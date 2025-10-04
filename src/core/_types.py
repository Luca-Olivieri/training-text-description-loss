from typing import Self, Any
from typing import TypeVar, Type, Generic, Optional, Iterable, Iterator
from typing import Callable, Literal, Generator, AsyncGenerator, Union
from typing_extensions import deprecated, override
from abc import ABC, abstractmethod
from PIL import Image
import torch

# Generics
T = TypeVar("T") # generic type
F = TypeVar("F", bound=Callable[..., object]) # generic function

TensorStructure = list[torch.Tensor, TypeVar("TensorStructure")] # nested list structure
StructureInfo = list[int | TypeVar("StructureInfo")] # tenso metadata structure

# Types for images
RGB_tuple = tuple[float, float, float]

# Types for prompts
Prompt = list[str | Image.Image]
Conversation = list[dict[str, str]] # list of chat-templated turns.

# Types for cache
TensorImage = torch.Tensor
TextInfo = str
CacheItem = tuple[TensorImage, TextInfo]
CacheKey = str

# Types for MLLMs
GenericClient = TypeVar['GenericClient']
