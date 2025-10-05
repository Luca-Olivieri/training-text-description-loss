from typing import Self, Any
from typing import TypeVar, Type, Generic, Optional, Iterable, Iterator, TypeAlias
from typing import Callable, Literal, Generator, AsyncGenerator, Union
from typing_extensions import deprecated, override
from abc import ABC, abstractmethod
from PIL import Image
import torch

from dataclasses import dataclass

# Generics
T = TypeVar("T") # generic type
K = TypeVar('K') # generic key type
V = TypeVar('V') # generic value type
F = TypeVar("F", bound=Callable[..., object]) # generic function

TensorStructure = list[torch.Tensor, TypeVar("TensorStructure")] # nested list structure
StructureInfo = list[int | TypeVar("StructureInfo")] # tenso metadata structure

# Types for images
RGB_tuple = tuple[float, float, float]

# Types for prompts
Prompt = list[str | Image.Image]
Conversation = list[dict[str, str]] # list of chat-templated turns.
PosClass = int
ClassSplitted = dict[PosClass, T]

# Types for cache
TensorImage = torch.Tensor
TextInfo = str
CacheItem = tuple[TensorImage, TextInfo]
CacheKey = str

# Types for MLLMs
GenericClient = TypeVar('GenericClient')
