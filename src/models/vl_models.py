from config import *
from data import append_many_to_jsonl, image_to_base64
from utils import DictObject, GenericClient, batch_list, my_tqdm, retry, batch_class_splitted_list, parse_eval_text_to_dict
from prompter import Prompt

from typing import Any, Generator, AsyncGenerator, Optional
from PIL.Image import Image as PILImage
import asyncio
from collections import OrderedDict
import time
import re

from abc import abstractmethod, ABC
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ServerError

from torch import nn
import torchmetrics as tm
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader

# Custom types
Conversation = list[dict[str, str]] # list of chat-templated turns.

# Ollama
import ollama

class GenParams(DictObject):
    """
    A dictionary-like class with a predefined set of keys, initially unassigned (None).
    The keys are defined within the class itself.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            ctx_size: Optional[int] = None,
            last_n_not_to_repeat: Optional[int] = None,
            repeat_penalty: Optional[float] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[list[str]] = None,
            max_tokens:  Optional[int]  = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            answer_format: Optional[str | dict[str, str]] = None
    ) -> None:
        """
        Initializea GenParams with optional generation parameters.

        Args:
            seed: Random seed for generation.
            ctx_size: Context window size.
            last_n_not_to_repeat: Number of last tokens not to repeat.
            repeat_penalty: Penalty for repeated tokens.
            temperature: Sampling temperature.
            stop_sequences: List of stop sequences.
            max_tokens: Maximum number of tokens to generate.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            min_p: Minimum probability for nucleus sampling.
            answer_format: Format for the answer.
        """
        self.seed = seed
        self.ctx_size = ctx_size
        self.last_n_not_to_repeat = last_n_not_to_repeat
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.stop_sequences = stop_sequences
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.answer_format = answer_format

class Response(DictObject):
    """
    Represents a model response, including the response text and optional token count.
    """
    def __init__(
            self,
            text: str,
            num_tokens: Optional[int] = None
    ) -> None:
        """
        Initializes a Response object.

        Args:
            text: The response text.
            num_tokens: Number of tokens in the response.
        """
        self.text = text
        self.num_tokens = num_tokens

class MLLM(ABC):
    """
    Base abstract class for implementing **Multimodal Large Language Models**.
    """
    def __init__(self) -> None:
        """
        Initializes the MLLM base class and set up the client.
        """
        self.client = self.get_client()
    
    def get_client(self) -> GenericClient:
        """
        Returns the client for the model. Override in subclasses.

        Returns:
            The client object or None.
        """
        return None
    
    @abstractmethod
    def convert_prompt_to_conv(
        self,
        user_prompt: list[str | PILImage],
        system_prompt: str,
    ) -> Conversation:
        """
        Converts user and system prompts to a conversation format.

        Args:
            user_prompt: List of user prompt strings or images.
            system_prompt: System prompt string.

        Returns:
            Conversation object.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def generate_response(
        self,
        conversation: Conversation,
        gen_params: GenParams,
        stream: bool = False
    )-> Response | AsyncGenerator[Response, None]:
        """
        Generates a response to a conversation.

        Args:
            conversation: The conversation object.
            gen_params: Generation parameters.
            stream: Whether to stream the response.

        Returns:
            Response object or async generator of responses.
        """
        raise NotImplementedError
    
    async def generate_response_batch(
            self,
            conversations: list[Prompt],
            gen_params: GenParams,
    ) -> list[Response]:
        """
        Generates responses for a batch of conversations.

        Args:
            conversations: List of conversation prompts.
            gen_params: Generation parameters.

        Returns:
            List of Response objects.
        """
        
        tasks = [self.generate_response(
            c, 
            gen_params=gen_params,
            stream=False
            )
            for c in conversations
        ]
        return await asyncio.gather(*tasks)
    
    async def predict_one(
            self,
            query_prompt: Prompt,
            query_idx: int,
            gen_params: GenParams,
            system_prompt: str = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
    ) -> tuple[int, Any]:
        """
        Predicts a response for a single prompt.

        Args:
            query_prompt: The prompt to predict for.
            query_idx: Index of the query.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.

        Returns:
            Dictionary with image index and content.
        """
        
        conv = self.convert_prompt_to_conv(query_prompt, system_prompt)
        
        answer_response: Response = await self.generate_response(
            conv,
            gen_params=gen_params,
            stream=False
        )
        
        out: Response = self.adapt_response(answer_response)
        out = self.process_response(out, only_text=only_text, parse_to_dict=parse_to_dict)

        return {"img_idx": query_idx, "content": out}
    
    async def _predict_batch(
            self,
            query_prompts: list[Prompt],
            query_idxs: list[int],
            gen_params: GenParams,
            system_prompt: Optional[str] = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
    ) -> list[tuple[int, Response]]:
        """
        Predicts responses for a batch of prompts.

        Args:
            query_prompts: List of prompts.
            query_idxs: List of query indices.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.

        Returns:
            List of dictionaries with image index and content.
        """
        
        convs = [self.convert_prompt_to_conv(
            inf_p,
            system_prompt)
            for inf_p in query_prompts]
        
        answer_responses = await self.generate_response_batch(
            convs,
            gen_params,
        )
        
        outs: Response = [self.adapt_response(a_resp) for a_resp in answer_responses]
        outs = [self.process_response(o, only_text=only_text, parse_to_dict=parse_to_dict) for o in outs]

        return [{"img_idx": i, "content": o} for i, o in zip(query_idxs, outs)]

    async def predict_many(
            self,
            query_prompts: list[Prompt],
            query_idxs: list[int],
            gen_params: GenParams,
            jsonl_save_path: Optional[Path],
            system_prompt: Optional[str] = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
            batch_size: Optional[int] = None,
            cooldown_period: float = 0.0
    ) -> list[tuple[int, Response]]:
        """
        Predicts responses for many prompts, optionally in batches.

        Args:
            query_prompts: List of prompts.
            query_idxs: List of query indices.
            gen_params: Generation parameters.
            jsonl_save_path: Path to save results as JSONL.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.
            batch_size: Batch size for processing.
            cooldown_period: Cooldown period between batches.

        Returns:
            List of dictionaries with image index and content.
        """
        epoch_answer_list = []
        if batch_size is not None:
            zipped_query_batches = batch_list(zip(query_idxs, query_prompts), batch_size)

            for step, batch in my_tqdm(zipped_query_batches):

                if step not in [0, len(zipped_query_batches)]: 
                    time.sleep(cooldown_period)
                img_idx_batch, eval_query_batch = zip(*batch)

                answer_pr_batch = await self._predict_batch(
                    eval_query_batch,
                    img_idx_batch,
                    gen_params=gen_params,
                    system_prompt=system_prompt,
                    only_text=only_text,
                    parse_to_dict=parse_to_dict
                )

                if jsonl_save_path is not None:
                    append_many_to_jsonl(jsonl_save_path, answer_pr_batch)

                epoch_answer_list.extend(answer_pr_batch)
        else:
            for _, (img_idx, q_p) in my_tqdm(zip(query_idxs, query_prompts)):  

                answer_pr = await self.predict_one(
                    q_p,
                    img_idx,
                    gen_params=gen_params,
                    system_prompt=system_prompt,
                    only_text=only_text,
                    parse_to_dict=parse_to_dict
                )
                
                if jsonl_save_path is not None:
                    append_many_to_jsonl(jsonl_save_path, [answer_pr])
                
                epoch_answer_list.append(answer_pr)

        return epoch_answer_list
    
    async def predict_one_class_splitted(
            self,
            class_splitted_query_prompt: dict[int, list[Prompt]],
            query_idx: int,
            gen_params: GenParams,
            system_prompt: str = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
            splits_in_parallel: bool = True,
    ) -> dict:
        """
        Predicts for a single prompt with class splits.

        Args:
            class_splitted_query_prompt: Dictionary of class to prompt list.
            query_idx: Index of the query.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.
            splits_in_parallel: Whether to process splits in parallel.

        Returns:
            Dictionary with image index and class-split content.
        """
        
        outs = {"img_idx": query_idx, "content": dict()}

        significant_classes = list(class_splitted_query_prompt.keys())
        query_prompts = list(class_splitted_query_prompt.values())

        if splits_in_parallel:
            class_splitted_answer_pr = await self._predict_batch(
                query_prompts,
                query_idxs=[query_idx]*len(significant_classes),
                gen_params=gen_params,
                system_prompt=system_prompt,
                only_text=only_text,
                parse_to_dict=parse_to_dict
            )

            outs["content"] = {c: a["content"] for c, a in zip(significant_classes, class_splitted_answer_pr)}
        else:
            for c, q_p in zip(significant_classes, query_prompts):
                class_splitted_answer_pr = await self.predict_one(
                    q_p,
                    query_idx=query_idx,
                    gen_params=gen_params,
                    system_prompt=system_prompt,
                    only_text=only_text,
                    parse_to_dict=parse_to_dict
                )

                outs["content"].update({c: class_splitted_answer_pr["content"]})
        
        return outs
    
    @retry(50, 60, [ServerError])
    async def _predict_batch_class_splitted(
            self,
            class_splitted_query_prompts: list[dict[int, list[Prompt]]],
            query_idxs: list[int],
            gen_params: GenParams,
            system_prompt: Optional[str] = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
            splits_in_parallel: bool = True
    ) -> list[tuple[int, Response]]:
        """
        Predicts for a batch of class-splitted prompts.

        Args:
            class_splitted_query_prompts: List of class-splitted prompts.
            query_idxs: List of query indices.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.
            splits_in_parallel: Whether to process splits in parallel.

        Returns:
            List of dictionaries with image index and class-split content.
        """
        tasks = [self.predict_one_class_splitted(
            q_p_splitted,
            query_idx=img_idx,
            gen_params=gen_params,
            system_prompt=system_prompt,
            only_text=only_text,
            parse_to_dict=parse_to_dict,
            splits_in_parallel=splits_in_parallel
        ) for img_idx, q_p_splitted in zip(query_idxs, class_splitted_query_prompts)]

        return await asyncio.gather(*tasks)
    
    async def predict_many_class_splitted(
            self,
            class_splitted_query_prompts: list[dict[int, list[Prompt]]],
            query_idxs: list[int],
            gen_params: GenParams,
            jsonl_save_path: Optional[Path],
            system_prompt: Optional[str] = None,
            only_text: bool = False,
            parse_to_dict: bool = False,
            splits_in_parallel: bool = True,
            batch_size: Optional[int] = None,
            cooldown_period: float = 0.0,
            use_tqdm: bool = True
    ) -> list[dict]:
        """
        Predicts for many class-splitted prompts, optionally in batches.

        Args:
            class_splitted_query_prompts: List of class-splitted prompts.
            query_idxs: List of query indices.
            gen_params: Generation parameters.
            jsonl_save_path: Path to save results as JSONL.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.
            splits_in_parallel: Whether to process splits in parallel.
            batch_size: Batch size for processing.
            cooldown_period: Cooldown period between batches.

        Returns:
            List of dictionaries with image index and class-split content.
        """

        my_tqdm_ = my_tqdm
        if not use_tqdm:
            my_tqdm_ = enumerate
        
        epoch_answer_list = []

        if batch_size is not None:
            zipped_query_batches = batch_class_splitted_list(zip(query_idxs, class_splitted_query_prompts), batch_size)

            for step, batch in my_tqdm_(zipped_query_batches):

                if step not in [0, len(zipped_query_batches)]:
                    time.sleep(cooldown_period)
                
                img_idx_batch, query_prompt_batch = zip(*batch)

                answer_pr_batch = await self._predict_batch_class_splitted(
                    query_prompt_batch,
                    img_idx_batch,
                    gen_params=gen_params,
                    only_text=only_text,
                    parse_to_dict=parse_to_dict,
                    splits_in_parallel=splits_in_parallel
                )

                if jsonl_save_path is not None:
                    append_many_to_jsonl(jsonl_save_path, answer_pr_batch)

                epoch_answer_list.extend(answer_pr_batch)
        else:
            for _, (img_idx, q_p_class_splitted) in my_tqdm_(zip(query_idxs, class_splitted_query_prompts)):  

                answer_pr = await self.predict_one_class_splitted(
                    q_p_class_splitted,
                    img_idx,
                    gen_params=gen_params,
                    system_prompt=system_prompt,
                    only_text=only_text,
                    parse_to_dict=parse_to_dict,
                    splits_in_parallel=splits_in_parallel
                )
                
                if jsonl_save_path is not None:
                    append_many_to_jsonl(jsonl_save_path, [answer_pr])

                epoch_answer_list.append(answer_pr)

        return epoch_answer_list
    
    async def evaluate_one_class_splitted(
            self,
            class_splitted_eval_prompt: dict[int, Prompt],
            query_idx: int,
            gen_params: GenParams,
            system_prompt: str = None,
            only_text: bool = False,
            parse_to_dict: Optional[bool] = False,
    ) -> dict:
        """
        Evaluates a single class-splitted prompt.

        Args:
            class_splitted_eval_prompt: Dictionary of class to prompt.
            query_idx: Index of the query.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.

        Returns:
            Dictionary with image index and evaluation content.
        """
        
        outs = {"img_idx": query_idx, "content": None}

        significant_classes = list(class_splitted_eval_prompt.keys())
        eval_prompts = list(class_splitted_eval_prompt.values())

        class_splitted_eval_pr = await self.evaluate_batch(
            eval_prompts,
            query_idxs=[query_idx]*len(class_splitted_eval_prompt),
            gen_params=gen_params,
            system_prompt=system_prompt,
            only_text=only_text,
            parse_to_dict=parse_to_dict
        )

        outs["content"] = {c: a["content"] for c, a in zip(significant_classes, class_splitted_eval_pr)}

        return outs
    
    async def evaluate_batch_class_splitted(
            self,
            class_splitted_eval_prompts: list[dict[int, Prompt]],
            query_idxs: int,
            gen_params: GenParams,
            system_prompt: str = None,
            only_text: bool = False,
            parse_to_dict: Optional[bool] = False,
    ) -> dict:
        """
        Evaluates a batch of class-splitted prompts.

        Args:
            class_splitted_eval_prompts: List of class-splitted evaluation prompts.
            query_idxs: List of query indices.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.

        Returns:
            List of dictionaries with image index and evaluation content.
        """
        
        tasks = [self.evaluate_one_class_splitted(
            e_p_splitted,
            query_idx=img_idx,
            gen_params=gen_params,
            system_prompt=system_prompt,
            only_text=only_text,
            parse_to_dict=parse_to_dict
        ) for img_idx, e_p_splitted in zip(query_idxs, class_splitted_eval_prompts)]

        return await asyncio.gather(*tasks)

    async def print_stream(
            self,
            stream_gen: AsyncGenerator[Response, None]
    ) -> None:
        """
        Prints a streamed response to stdout.

        Args:
            stream_gen: Async generator of Response objects.
        """
        async for chunk in stream_gen:
            print(chunk["message"]["content"], end='', flush=True)
        print()
    
    @abstractmethod
    def adapt_response(
            self,
            response: Any
    ) -> Response:
        """
        Adapts a raw model response to a Response object.

        Args:
            response: The raw response from the model.

        Returns:
            Adapted Response object.
        """
        raise NotImplementedError

    def process_response(
            self,
            response: Response,
            only_text: bool = False,
            parse_to_dict: bool = False
    ) -> str | dict[str, Any]:
        """
        Processes a Response object, optionally extracting only text or parsing to dict.

        Args:
            response: The Response object.
            only_text: Whether to return only text.
            parse_to_dict: Whether to parse output to dict.

        Returns:
            Processed response as string or dictionary.
        """
        if only_text is False and parse_to_dict is True:
            print("WARNING: 'parse_to_dict' is True but not applied because 'only_text' is False.")
        out = response
        if only_text is True:
            out: str = out.text
            out: dict[str, Any] = parse_eval_text_to_dict(out) if parse_to_dict else out
        return out 
    
class OllamaMLLM(MLLM):
    """
    Multimodal Large Language Model implementation for Ollama backend.
    Handles model initialization and client setup for Ollama models.
    """
    def __init__(
            self,
            model_name: str,
            container_name: str = "olivieri_ollama"
    ) -> None:
        """
        Initializes OllamaMLLM with model name and optional client.

        Args:
            model_name: Name of the model.
            client: Optional Ollama client.
        """
        self.model = model_name
        self.client = self.get_client(host=f"http://{container_name}:11434", async_=True)

    def get_client(
            self,
            host: str = "http://olivieri_ollama:11434",
            async_: bool = False
    ) -> ollama.Client:
        """
        Gets an Ollama client instance.

        Args:
            host: Host address for the Ollama server.
            async_: Whether to use the async client.

        Returns:
            Ollama client instance.
        """
        client = ollama.Client(host=host) if not async_ else ollama.AsyncClient(host=host)
        return client
    
    def convert_prompt_to_conv(
            self,
            user_prompt: Prompt,
            system_prompt: str
    ) -> Conversation:
        """
        Converts user and system prompts to Ollama conversation format.

        Args:
            user_prompt: User prompt (string or list).
            system_prompt: System prompt string.

        Returns:
            Conversation object for Ollama.
        """

        images_base64 = [image_to_base64(item) for item in user_prompt if isinstance(item, PILImage)] # extract list of images from prompt

        if isinstance(user_prompt, str) and len(images_base64) != 0:
            raise ValueError(f"If the user prompt is of type 'str', there must be no images: instead there are {len(images_base64)} images.")
        if isinstance(user_prompt, list):
            user_prompt = ["[img]" if isinstance(item, PILImage) else item for item in user_prompt] # replaces all images with "[img]"
        if any([not isinstance(piece, str) for piece in user_prompt]):
            print(user_prompt)
            raise TypeError(f"After having substitued PIL images with the placeholder '[img]', some pieces are not of type 'str'.")
        if isinstance(user_prompt, list):
            user_prompt = "".join(user_prompt)
        if not isinstance(user_prompt, str):
            raise TypeError(f"user prompt content fed to LMML must be of type 'str', it is of type '{type(user_prompt)}'.")
        user_prompt = re.sub(r'(\[img\]+)(?!\[img\]|$)', r'\1\n', user_prompt)

        conversation = []
        if system_prompt is not None:
            conversation.append({"role": "system_prompt", "content": system_prompt}) # add system turn if present
        conversation.append({"role": "user", "content": user_prompt, "images": images_base64}) # add user turn

        return conversation
    
    async def generate_response(
            self,
            conversation: Conversation,
            gen_params: GenParams,
            stream: bool = False,
    ) -> ollama.ChatResponse | Generator[ollama.ChatResponse, None, ollama.ChatResponse] | AsyncGenerator[ollama.ChatResponse, None]:
        """
        Generates a response using Ollama.

        Args:
            conversation: Conversation object for Ollama.
            gen_params: Generation parameters.
            stream: Whether to stream the response.

        Returns:
            Ollama ChatResponse or generator/async generator of responses.
        """
        
        response_or_generator: ollama.ChatResponse = await self.client.chat(
            model=self.model, 
            messages=conversation,
            options={
                "num_ctx": gen_params["ctx_size"] ,
                "repeat_last_n": gen_params["last_n_not_to_repeat"],
                "repeat_penalty": gen_params["repeat_penalty"],
                "temperature": gen_params["temperature"],
                "seed": gen_params["seed"],
                "stop": gen_params["stop_sequences"],
                "num_predict": gen_params["max_tokens"],
                "top_k": gen_params["top_k"],
                "top_p": gen_params["top_p"],
                "min_p": gen_params["min_p"],
            },
            format=gen_params["answer_format"],
            stream=stream
        ) # generate response or generator of responses

        return response_or_generator    
    
    def adapt_response(
            self,
            response: ollama.ChatResponse
    ) -> Response:
        """
        Adapts an Ollama ChatResponse to a Response object.

        Args:
            response: Ollama ChatResponse object.

        Returns:
            Adapted Response object.
        """
        new_response: Response = Response(
            text=response["message"]["content"],
            num_tokens=None
        )
        return new_response

class GoogleAIStudioMLLM(MLLM):
    def __init__(
            self,
            model_name: str,
            client: Optional[genai.Client] = None,
            api_key: str = None
    ) -> None:
        """
        Initializes GoogleAIStudioMLLM with model name, client, and API key.

        Args:
            model_name: Name of the model.
            client: Optional Google GenAI client.
            api_key: API key for authentication.
        """
        self.model = model_name
        if client is not None and api_key is not None:
            print("WARNING: 'api_key' is specified but unused because the client has been set.")
        self.client = client or self.get_client(api_key=api_key)

    def get_client(
            self,
            api_key: str
    ) -> genai.Client:
        """
        Gets a Google GenAI client instance.

        Args:
            api_key: API key for authentication.

        Returns:
            Google GenAI client instance.
        """
        client = genai.Client(api_key=api_key)
        return client
    
    def convert_prompt_to_conv(
            self,
            user_prompt: Prompt,
            system_prompt: str
    ) -> Prompt:
        """
        Converts user and system prompts to Google GenAI format.

        Args:
            user_prompt: User prompt.
            system_prompt: System prompt string.

        Returns:
            Prompt for Google GenAI.
        """
        return user_prompt

    async def generate_response(
            self,
            user_prompt: Prompt,
            gen_params: GenParams,
            system_prompt: Optional[str] = None,
            stream: bool = False
    ) -> genai_types.GenerateContentResponse | AsyncGenerator[genai_types.GenerateContentResponse, None]:    
        """
        Generates a response using Google GenAI.

        Args:
            user_prompt: Prompt for Google GenAI.
            gen_params: Generation parameters.
            system_prompt: Optional system prompt.
            stream: Whether to stream the response.

        Returns:
            Google GenAI response or async generator of responses.
        """
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                maxOutputTokens=gen_params["max_tokens"],
                stop_sequences=gen_params["stop_sequences"],
                seed=CONFIG["seed"]
            ),
        )

        return response

    def adapt_response(
            self,
            response: genai_types.GenerateContentResponse
    ) -> Response:
        """
        Adapts a Google GenAI response to a Response object.

        Args:
            response: Google GenAI response object.

        Returns:
            Adapted Response object.
        """
        new_response: Response = Response(
            text=response.text,
            num_tokens=None
        )
        return new_response
    
class HuggingFaceMLLM(MLLM):
    """
    Multimodal Large Language Model implementation for HuggingFace backend.
    Handles model initialization for HuggingFace models.
    """
    def __init__(
            self,
            model_name: str,
    ) -> None:
        """
        Initialize HuggingFaceMLLM with model name.

        Args:
            model_name: Name of the model.
        """
        self.model = model_name
        pass

def main() -> None:
    pass
    
if __name__ == '__main__':
    main()
