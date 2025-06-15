from config import *
from data import *
from prompter import PromptBuilder
from utils import *

import os
from typing import Any, Dict, Generator, AsyncGenerator, List, TypeVar, Optional
from PIL import Image
from PIL.Image import Image as PILImage
import asyncio

# Google
from google import genai
from google.genai import types as genai_types

from google.genai.errors import ServerError

# Ollama
import ollama

class GenParams(DictObject):
    """
    A dictionary-like class with a predefined set of keys, initially unassigned (None).
    The keys are defined within the class itself.
    """
    # TODO: comment each key.
    def __init__(
            self,
            seed: Optional[int] = None,
            ctx_size: Optional[int] = None,
            last_n_not_to_repeat: Optional[int] = None,
            repeat_penalty: Optional[float] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            max_tokens:  Optional[int]  = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            answer_format: Optional[str | Dict[str, str]] = None
    ) -> None:
        """
        Initializes the `GenParams` object with the predefined keys, all set to None.
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
    TODO
    """
    def __init__(
            self,
            text: str,
            num_tokens: Optional[int] = None
    ) -> None:
        self.text = text
        self.num_tokens = num_tokens

class MLLM(ABC):
    """
    Base abstract class for implementing **Multimodal Large Language Models**.
    """
    def __init__(self) -> None:
        self.client = self.get_client()
    
    def get_client(self) -> GenericClient:
        return None
    
    @abstractmethod
    def convert_prompt_to_conv(
        self,
        user_prompt: list[str | PILImage],
        system_prompt: str,
    ) -> Conversation:
        raise NotImplementedError
    
    @abstractmethod
    async def generate_response(
        self,
        conversation: Conversation,
        gen_params: GenParams,
        stream: bool = False
    )-> Response | AsyncGenerator[Response, None]:
        """
        Let the model generate an answer to a given user prompt.
        """
        raise NotImplementedError
    
    async def generate_response_batch(
            self,
            conversations: list[Prompt],
            gen_params: GenParams,
    ) -> list[Response]:
        
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
        
        conv = self.convert_prompt_to_conv(query_prompt, system_prompt)
        
        answer_response: GenericResponse = await self.generate_response(
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
    ) -> list[tuple[int, GenericResponse]]:
        
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
    ) -> list[tuple[int, GenericResponse]]:
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
    ) -> list[tuple[int, GenericResponse]]:
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
            cooldown_period: float = 0.0
    ) -> list[dict]:
        
        epoch_answer_list = []

        if batch_size is not None:
            zipped_query_batches = batch_class_splitted_list(zip(query_idxs, class_splitted_query_prompts), batch_size)

            for step, batch in my_tqdm(zipped_query_batches):

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
            for _, (img_idx, q_p_class_splitted) in my_tqdm(zip(query_idxs, class_splitted_query_prompts)):  

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
        
        tasks = [self.evaluate_one_class_splitted(
            e_p_splitted,
            query_idx=img_idx,
            gen_params=gen_params,
            system_prompt=system_prompt,
            only_text=only_text,
            parse_to_dict=parse_to_dict
        ) for img_idx, e_p_splitted in zip(query_idxs, class_splitted_eval_prompts)]

        return await asyncio.gather(*tasks)

    async def print_stream(self, stream_gen: AsyncGenerator[Response, None]) -> None:
        async for chunk in stream_gen:
            print(chunk["message"]["content"], end='', flush=True)
        print()
    
    @abstractmethod
    def adapt_response(self, response: Any):
        raise NotImplementedError

    def process_response(self, response: Response, only_text: bool = False, parse_to_dict: bool = False):
        if only_text is False and parse_to_dict is True:
            print("WARNING: 'parse_to_dict' is True but not applied because 'only_text' is False.")
        out = response
        if only_text is True:
            out: str = out.text
            out: dict[str, Any] = parse_eval_text_to_dict(out) if parse_to_dict else out
        return out 
    
class OllamaMLLM(MLLM):
    """
    TODO
    """
    def __init__(
            self,
            model_name: str,
            client: Optional[ollama.AsyncClient] = None
    ) -> None:
        self.model = model_name
        self.client = client or self.get_client(async_=True)

    def get_client(
            self,
            host="http://olivieri_ollama:11434",
            async_=False
    ) -> ollama.Client:
        client = ollama.Client(host=host) if not async_ else ollama.AsyncClient(host=host)
        return client
    
    def convert_prompt_to_conv(
            self,
            user_prompt: Prompt,
            system_prompt: str
    ) -> Conversation:

        images_base64 = [image_to_base64(item) for item in user_prompt if isinstance(item, PILImage)] # extract list of images from prompt

        if isinstance(user_prompt, str) and len(images_base64) != 0:
            raise ValueError(f"If the user prompt is of type 'str', there must be no images: instead there are {len(images_base64)} images.")
        if isinstance(user_prompt, list):
            user_prompt = ["[img]" if isinstance(item, PILImage) else item for item in user_prompt] # replaces all images with "[img]"
        if any([not isinstance(piece, str) for piece in user_prompt]):
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
    
    def adapt_response(self, response: ollama.ChatResponse):
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
        self.model = model_name
        if client is not None and api_key is not None:
            print("WARNING: 'api_key' is specified but unused because the client has been set.")
        self.client = client or self.get_client(api_key=api_key)

    def get_client(self, api_key: str) -> genai.Client:
        client = genai.Client(api_key=api_key)
        return client
    
    def convert_prompt_to_conv(
            self,
            user_prompt: Prompt,
            system_prompt: str
    ) -> Prompt:
        return user_prompt

    async def generate_response(
            self,
            user_prompt: Prompt,
            gen_params: GenParams,
            system_prompt: Optional[str] = None,
            stream: bool = False
    ) -> genai_types.GenerateContentResponse | AsyncGenerator[genai_types.GenerateContentResponse, None]:    

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
                seed=get_seed()
            ),
        )

        return response

    def adapt_response(self, response: genai_types.GenerateContentResponse):
        new_response: Response = Response(
            text=response.text,
            num_tokens=None
        )
        return new_response
    
class HuggingFaceMLLM(MLLM):
    """
    TODO
    """
    def __init__(
            self,
            model_name: str,
    ) -> None:
        self.model = model_name
        pass

def main() -> None:
    mllm = OllamaMLLM(model_name="gemma3:4b-it-qat")

    image_paths = ["/home/olivieri/exp/random1.png", "/home/olivieri/exp/random2.png"]
    # image_paths = ["/home/olivieri/exp/random1.png", "
