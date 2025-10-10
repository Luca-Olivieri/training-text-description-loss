"""Multimodal Large Language Model (MLLM) adapters and interfaces.

This module provides a unified interface for interacting with various MLLM backends through
abstract base classes and concrete implementations. It supports async inference, batch processing,
and multimodal prompts (text + images) across different model providers.

Supported backends:
    - Ollama: Local/self-hosted models (Gemma, Qwen2.5-VL, InternVL3, etc.)
    - Google AI Studio: Gemini models via GenAI API
    - HuggingFace: Placeholder for future Transformers integration

Key components:
    - MLLMAdapter: Abstract base class defining the MLLM interface
    - MLLMGenParams: Frozen dataclass for generation hyperparameters
    - MLLMResponse: Response container with text and optional token count
    - OllamaMLLMAdapter: Ollama backend with base64 image encoding
    - GoogleAIStudioMLLMAdapter: Google Gemini backend with retry logic
    - MLLM_REGISTRY: Global registry for adapter factory functions

Usage example:
    >>> adapter = MLLM_REGISTRY.get('gemini-2.0-flash')(api_key='...')
    >>> gen_params = MLLMGenParams(temperature=0.7, max_tokens=512)
    >>> response = await adapter.predict_one(prompt, gen_params)

TODO:
    - Remove class-splitting notion from MLLM using Grouped Sampler approach
"""
from __future__ import annotations

from core.config import *
from core.data import image_to_base64
from core.prompter import Prompt
from core.registry import Registry
from core.utils import retry

import asyncio
import re
from PIL import Image
from functools import partial

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

from core._types import Optional, Conversation, abstractmethod, ABC, dataclass

# Ollama
import ollama

@dataclass(frozen=True)
class MLLMGenParams:
    """Generation parameters for MLLM inference.
    
    Encapsulates all generation hyperparameters used to control the behavior
    of multimodal large language models during text generation.
    
    Attributes:
        seed: Random seed for reproducible generation. None for non-deterministic.
        ctx_size: Maximum context window size (number of tokens).
        last_n_not_to_repeat: Number of last tokens to avoid repeating.
        repeat_penalty: Penalty multiplier for repeated tokens (>1.0 reduces repetition).
        temperature: Sampling temperature (higher = more random, 0.0 = greedy).
        stop_sequences: List of strings that stop generation when encountered.
        max_tokens: Maximum number of tokens to generate in the response.
        top_k: Top-k sampling - only sample from top k most likely tokens.
        top_p: Nucleus sampling - sample from smallest set with cumulative probability >= top_p.
        min_p: Minimum probability threshold for token consideration.
        answer_format: Expected format for the answer ('json', dict schema, or None).
    """
    seed: Optional[int] = None
    ctx_size: Optional[int] = None
    last_n_not_to_repeat: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    max_tokens:  Optional[int]  = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    answer_format: Optional[str | dict[str, str]] = None

# TODO Generate text in the dict default format, not in the JSONL format.

@dataclass()
class MLLMResponse:
    """Container for MLLM generation response.
    
    Encapsulates the text output from an MLLM along with optional metadata
    about token usage.
    
    Attributes:
        text: The generated response text from the model.
        num_tokens: Optional count of tokens in the response (None if not provided by backend).
    """
    text: str
    num_tokens: Optional[int] = None

# TODO the JSONL saving logic for the MLLMs is embedded in the text generation logic, decouple it.

# TODO save the text predictions as separate objects (like for the synthetic diff dataset), the MLLM should consider as individual samples the positive samples (not the variable-sized class-splitted ones). Then, when we need to fetch whole cs-splitted elements used the Grouped Sampler approach (see 'https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221kDHqSTnRIaWOO7nN4CD9GVARKI44IsXK%22%5D,%22action%22:%22open%22,%22userId%22:%22101546796642867554797%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing'). The MLLM should have no notion of class-splitting.

class MLLMAdapter(ABC):
    """Abstract base class for Multimodal Large Language Model adapters.
    
    Defines the interface that all MLLM backend implementations must follow.
    Subclasses should implement model-specific client initialization, prompt
    preprocessing, and response generation logic.
    
    Subclasses must implement:
        - __init__: Initialize the MLLM client and model
        - predict_one: Generate response for a single prompt
        - predict_batch: Generate responses for multiple prompts
    """
    @abstractmethod
    def __init__(self) -> None:
        """Initialize the MLLM adapter and set up the backend client.
        
        Must be implemented by subclasses to configure model-specific
        clients, authentication, and endpoints.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    async def predict_one(
            self,
            prompt: Prompt,
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> MLLMResponse:
        """Generate a response for a single prompt (async).
        
        Args:
            prompt: The input prompt (can be text, list of text/images).
            gen_params: Generation parameters controlling model behavior.
            system_prompt: Optional system instruction to guide model behavior.
        
        Returns:
            MLLMResponse containing the generated text and token count.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def predict_batch(
            self,
            prompts: list[Prompt],
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> list[MLLMResponse]:
        """Generate responses for a batch of prompts concurrently (async).
        
        Processes multiple prompts in parallel for efficient batch inference.
        All prompts share the same generation parameters and system prompt.

        Args:
            prompts: List of input prompts (each can be text or list of text/images).
            gen_params: Generation parameters applied to all prompts.
            system_prompt: Optional system instruction applied to all prompts.

        Returns:
            List of MLLMResponse objects, one for each input prompt in order.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    

MLLM_REGISTRY = Registry[MLLMAdapter]()
"""Global registry mapping model names to MLLM adapter factory functions.

This registry enables dynamic adapter instantiation by model name. Models are
registered using partial functions that bind model-specific parameters. Access
adapters using MLLM_REGISTRY.get(model_name) which returns a factory function
that creates the appropriate adapter instance.
"""


class OllamaMLLMAdapter(MLLMAdapter):
    """Ollama backend adapter for multimodal large language models.
    
    Provides integration with Ollama's local/self-hosted MLLM backend.
    Handles conversation formatting, image encoding to base64, and
    asynchronous response generation using the Ollama API.
    
    Attributes:
        model: Name of the Ollama model to use.
        http_endpoint: HTTP endpoint URL for the Ollama server.
        client: Async Ollama client instance.
    """
    def __init__(
            self,
            model_name: str,
            http_endpoint: str
    ) -> None:
        """Initialize the Ollama MLLM adapter.

        Args:
            model_name: Name of the Ollama model (e.g., 'llava', 'gemma3:4b-it').
            http_endpoint: HTTP endpoint URL for the Ollama server (e.g., 'http://localhost:11434').
        """
        self.model = model_name
        self.http_endpoint = http_endpoint
        self.client = self._create_client(http_endpoint=http_endpoint, async_=True)

    async def predict_one(
            self,
            prompt: Prompt,
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> MLLMResponse:
        """Generate a response for a single prompt using Ollama.

        Preprocesses the prompt into Ollama's conversation format (including
        base64-encoded images) and generates a response asynchronously.

        Args:
            prompt: Input prompt (string or list of strings/PIL.Image objects).
            gen_params: Generation parameters controlling model behavior.
            system_prompt: Optional system instruction to guide the model.

        Returns:
            MLLMResponse containing the generated text.
        """
        conv = self._preprocess_prompt(prompt, system_prompt)
        
        response: MLLMResponse = await self._generate_response(conv, gen_params=gen_params)

        return response  

    async def predict_batch(
            self,
            prompts: list[Prompt],
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> list[MLLMResponse]:
        """Generate responses for a batch of prompts concurrently using Ollama.

        Preprocesses all prompts into Ollama's conversation format and
        generates responses in parallel using asyncio.gather.

        Args:
            prompts: List of input prompts (each can be string or list of strings/PIL.Image objects).
            gen_params: Generation parameters applied to all prompts.
            system_prompt: Optional system instruction applied to all prompts.

        Returns:
            List of MLLMResponse objects, one for each input prompt in order.
        """
        convs: list[Conversation] = [self._preprocess_prompt(prompt, system_prompt) for prompt in prompts]

        responses: list[MLLMResponse] = await self._generate_response_batch(convs, gen_params)

        return responses
    
    async def _generate_response(
            self,
            conversation: Conversation,
            gen_params: MLLMGenParams,
    ) -> MLLMResponse:
        """Generate a response using the Ollama chat API.
        
        Internal method that calls the Ollama client with formatted conversation
        and generation parameters.

        Args:
            conversation: Formatted conversation with role/content/images structure.
            gen_params: Generation parameters (temperature, top_p, seed, etc.).

        Returns:
            MLLMResponse containing the generated text.
        """
        ollama_response: ollama.ChatResponse = await self.client.chat(
            model=self.model, 
            messages=conversation,
            options={
                "num_ctx": gen_params.ctx_size,
                "repeat_last_n": gen_params.last_n_not_to_repeat,
                "repeat_penalty": gen_params.repeat_penalty,
                "temperature": gen_params.temperature,
                "seed": gen_params.seed,
                "stop": gen_params.stop_sequences,
                "num_predict": gen_params.max_tokens,
                "top_k": gen_params.top_k,
                "top_p": gen_params.top_p,
                "min_p": gen_params.min_p,
            },
            format=gen_params.answer_format,
        )

        response: MLLMResponse = self._adapt_response(ollama_response)

        return response

    async def _generate_response_batch(
            self,
            conversations: list[Conversation],
            gen_params: MLLMGenParams,
    ) -> list[MLLMResponse]:
        """Generate responses for multiple conversations concurrently.
        
        Internal method that creates async tasks for each conversation and
        gathers results in parallel.

        Args:
            conversations: List of formatted conversations (role/content/images).
            gen_params: Generation parameters applied to all conversations.

        Returns:
            List of MLLMResponse objects in the same order as input conversations.
        """
        tasks = [self._generate_response(conv, gen_params=gen_params) for conv in conversations]
        responses: list[MLLMResponse] = await asyncio.gather(*tasks)
        return responses
    
    def _create_client(
            self,
            http_endpoint: str,
            async_: bool = True
    ) -> ollama.Client | ollama.AsyncClient:
        """Create an Ollama client instance.
        
        Factory method to instantiate either sync or async Ollama client.

        Args:
            http_endpoint: HTTP endpoint URL for the Ollama server (e.g., 'http://localhost:11434').
            async_: Whether to create an async client (True) or sync client (False).

        Returns:
            ollama.AsyncClient if async_ is True, otherwise ollama.Client.
        """
        if async_:
            client = ollama.AsyncClient(host=http_endpoint)
        else:
            client = ollama.Client(host=http_endpoint)
        return client
    
    
    # TODO split the internal logic of this method into simpler functions.
    def _preprocess_prompt(
            self,
            user_prompt: Prompt,
            system_prompt: Optional[str] = None
    ) -> Conversation:
        """Convert user and system prompts to Ollama conversation format.
        
        Processes the input prompt by:
        1. Extracting PIL images and encoding them to base64
        2. Replacing image objects with '[img]' placeholders
        3. Joining text pieces into a single string
        4. Adding newlines between images and text
        5. Formatting as Ollama conversation structure

        Args:
            user_prompt: User prompt (string, or list of strings and PIL.Image objects).
            system_prompt: Optional system instruction string.

        Returns:
            Conversation list with 'system' (if provided) and 'user' turns,
            where user turn includes content string and base64-encoded images.
        
        Raises:
            ValueError: If string prompt contains images.
            TypeError: If prompt contains non-string, non-image elements after processing.
        """

        images_base64 = [image_to_base64(item) for item in user_prompt if isinstance(item, Image.Image)] # extract list of images from prompt

        if isinstance(user_prompt, str) and len(images_base64) != 0:
            raise ValueError(f"If the user prompt is of type 'str', there must be no images: instead there are {len(images_base64)} images.")
        if isinstance(user_prompt, list):
            user_prompt = ["[img]" if isinstance(item, Image.Image) else item for item in user_prompt] # replaces all images with "[img]"
        if any([not isinstance(piece, str) for piece in user_prompt]):
            print(user_prompt)
            raise TypeError(f"After having substitued PIL images with the placeholder '[img]', some pieces are not of type 'str'.")
        if isinstance(user_prompt, list):
            user_prompt = "".join(user_prompt)
        if not isinstance(user_prompt, str):
            raise TypeError(f"user prompt content fed to MLLM must be of type 'str', it is of type '{type(user_prompt)}'.")
        user_prompt = re.sub(r'(\[img\]+)(?!\[img\]|$)', r'\1\n', user_prompt) # adds a '\n' between an image if it is followed by text.

        conversation = []
        if system_prompt:
            conversation.append({'role': 'system', 'content': system_prompt}) # add system turn if present
        conversation.append({'role': 'user', 'content': user_prompt, 'images': images_base64}) # add user turn

        return conversation
    
    def _adapt_response(
            self,
            ollama_response: ollama.ChatResponse
    ) -> MLLMResponse:
        """Convert Ollama ChatResponse to MLLMResponse format.
        
        Extracts the message content from Ollama's response structure.

        Args:
            ollama_response: Raw response dictionary from Ollama chat API.

        Returns:
            MLLMResponse containing the message content text.
        """
        response: MLLMResponse = MLLMResponse(
            text=ollama_response["message"]["content"],
            num_tokens=None
        )
        return response
    
    async def load_model(self) -> None:
        """Pre-load the model into memory by making a dummy prediction.
        
        Sends an empty prompt to warm up the model and load it into memory.
        Useful for reducing latency on the first real prediction.
        """
        await self.predict_one([""], MLLMGenParams())


valid_ollama_model_names: list[str] = [
    'gemma3:4b-it-q4_K_M',
    'gemma3:4b-it-qat',
    'gemma3:12b',
    'gemma3:12b-it-qat',
    'gemma3:27b',
    'gemma3:27b-it-qat',
    'qwen2.5vl:7b-q4_K_M',
    'qwen2.5vl:7b-q8_0',
    'hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M',
    'hf.co/unsloth/InternVL3-14B-Instruct-GGUF:Q4_K_M'
]
"""List of validated Ollama model names supported by the adapter.

Includes various quantization levels (Q4_K_M, Q8_0) and model sizes (4B-27B) 
for Gemma3, Qwen2.5-VL, Mistral-Small, and InternVL3 models available through Ollama.
"""

# Register all valid Ollama models in the global MLLM_REGISTRY
# Each model is registered with a partial factory function that binds the model_name parameter
for model_name in valid_ollama_model_names:
    MLLM_REGISTRY.add(model_name, partial(OllamaMLLMAdapter, model_name=model_name))


class GoogleAIStudioMLLMAdapter(MLLMAdapter):
    """Google AI Studio backend adapter for multimodal large language models.
    
    Provides integration with Google's Gemini models through the GenAI API.
    Handles native multimodal prompt formatting and asynchronous response
    generation using Google's API client.
    
    Attributes:
        model: Name of the Google model to use (e.g., 'gemini-2.0-flash').
        client: Google GenAI client instance configured with API key.
    """
    def __init__(
            self,
            model_name: str,
            api_key: str
    ) -> None:
        """Initialize the Google AI Studio MLLM adapter.

        Args:
            model_name: Name of the Google model (e.g., 'gemini-2.0-flash').
            api_key: Google API key for authentication and access.
        """
        self.model = model_name
        self.client = self._create_client(api_key=api_key)

    async def predict_one(
            self,
            prompt: Prompt,
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> MLLMResponse:
        """Generate a response for a single prompt using Google AI Studio.

        Args:
            prompt: The input prompt (text, list of text/images, or native Google format).
            gen_params: Generation parameters controlling model behavior.
            system_prompt: Optional system instruction to guide the model.

        Returns:
            MLLMResponse containing the generated text and token count.
        """
        response: MLLMResponse = await self._generate_response(prompt, gen_params=gen_params, system_prompt=system_prompt)

        return response

    @retry(max_retries=3, cooldown_period=60, exceptions=[genai_errors.APIError])
    async def predict_batch(
            self,
            prompts: list[Prompt],
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> list[MLLMResponse]:
        """Generate responses for a batch of prompts concurrently using Google AI Studio.
        
        Processes multiple prompts in parallel using asyncio.gather for efficient
        batch inference. Automatically retries on API errors (up to 3 times with
        60-second cooldown between retries).

        Args:
            prompts: List of input prompts (each can be text, list of text/images, or native format).
            gen_params: Generation parameters applied to all prompts.
            system_prompt: Optional system instruction applied to all prompts.

        Returns:
            List of MLLMResponse objects, one for each input prompt in order.
        
        Raises:
            genai_errors.APIError: If API errors persist after max retries.
        """
        responses: list[MLLMResponse] = await self._generate_response_batch(prompts, gen_params, system_prompt)

        return responses
    
    def _create_client(
            self,
            api_key: str
    ) -> genai.Client:
        """Create a Google GenAI client instance.
        
        Factory method to instantiate the Google GenAI client with authentication.

        Args:
            api_key: Google API key for authentication and authorization.

        Returns:
            Configured genai.Client instance ready for API calls.
        """
        client = genai.Client(api_key=api_key)
        return client

    async def _generate_response(
            self,
            user_prompt: Prompt,
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> MLLMResponse:
        """Generate a response using the Google GenAI API.
        
        Internal method that calls the Google GenAI client with the prompt
        and generation configuration.

        Args:
            user_prompt: Input prompt in Google's native format (supports multimodal).
            gen_params: Generation parameters (temperature, top_p, max_tokens, etc.).
            system_prompt: Optional system instruction passed as system_instruction.

        Returns:
            MLLMResponse containing the generated text.
        """
        genai_response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                top_k=gen_params.top_k,
                maxOutputTokens=gen_params.max_tokens,
                stop_sequences=gen_params.stop_sequences,
                seed=gen_params.seed
            ),
        )

        response: MLLMResponse = self._adapt_response(genai_response)

        return response

    async def _generate_response_batch(
            self,
            prompts: list[Prompt],
            gen_params: MLLMGenParams,
            system_prompt: Optional[str] = None,
    ) -> list[MLLMResponse]:
        """Generate responses for multiple prompts concurrently.
        
        Internal method that creates async tasks for each prompt and
        gathers results in parallel.

        Args:
            prompts: List of input prompts in Google's native format.
            gen_params: Generation parameters applied to all prompts.
            system_prompt: Optional system instruction applied to all prompts.

        Returns:
            List of MLLMResponse objects in the same order as input prompts.
        """
        tasks = [self._generate_response(prompt, gen_params=gen_params, system_prompt=system_prompt) for prompt in prompts]
        responses: list[MLLMResponse] = await asyncio.gather(*tasks)
        return responses

    def _adapt_response(
            self,
            genai_response: genai_types.GenerateContentResponse
    ) -> MLLMResponse:
        """Convert Google GenAI response to MLLMResponse format.
        
        Extracts the text content from Google's response structure.

        Args:
            genai_response: Raw GenerateContentResponse from Google GenAI API.

        Returns:
            MLLMResponse containing the response text.
        """
        response: MLLMResponse = MLLMResponse(
            text=genai_response.text,
            num_tokens=None
        )
        return response
    
valid_googleaistudio_model_names: list[str] = [
    'gemini-2.0-flash',
]
"""List of validated Google AI Studio model names supported by the adapter.

Currently includes Gemini 2.0 Flash model available through Google's GenAI API.
"""

# Register all valid Google AI Studio models in the global MLLM_REGISTRY
# Each model is registered with a partial factory function that binds the model_name parameter
for model_name in valid_googleaistudio_model_names:
    MLLM_REGISTRY.add(model_name, partial(GoogleAIStudioMLLMAdapter, model_name=model_name))

class HuggingFaceMLLM(MLLMAdapter):
    """HuggingFace backend adapter for multimodal large language models.
    
    Placeholder for future implementation of HuggingFace Transformers integration.
    Will provide support for locally-hosted or HF-hosted multimodal models.
    
    TODO: Implement HuggingFace adapter with:
        - Model loading from HuggingFace Hub
        - Local inference support
        - Processor/tokenizer handling
        - Image preprocessing pipeline
    """
    ...

def main() -> None:
    """Main entry point for the module.
    
    Currently a placeholder. Can be used for testing MLLM adapters
    or running example inference.
    """
    pass
    
if __name__ == '__main__':
    main()
