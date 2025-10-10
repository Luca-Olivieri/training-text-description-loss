from typing import Self, Any
from typing import TypeVar, Type, Generic, Optional, Iterable, Iterator, TypeAlias
from typing import Callable, Literal, Generator, AsyncGenerator, Union
from typing_extensions import deprecated, override
from abc import ABC, abstractmethod
from PIL import Image
import torch

from dataclasses import dataclass

# Generics
T = TypeVar('T') # generic type
R = TypeVar('R') # return generic type
K = TypeVar('K') # generic key type
V = TypeVar('V') # generic value type
F = TypeVar('F', bound=Callable[..., object]) # generic function

TensorStructureInfo: TypeAlias = list[torch.Tensor, 'TensorStructureInfo'] # nested list structure
ListStructureInfo: TypeAlias = list[int, 'ListStructureInfo'] # tenso metadata structure

# Types for images
RGB_tuple: TypeAlias = tuple[float, float, float]
ImageMediaMetadata: TypeAlias = object

# Types for prompts
Prompt: TypeAlias = list[str | Image.Image]
Conversation: TypeAlias = list[dict[str, str]] # list of chat-templated turns.
PosClass: TypeAlias = int
ClassSplitted: TypeAlias = dict[PosClass, T]

# Types for cache
TensorImage: TypeAlias = torch.Tensor
TextInfo: TypeAlias = str
CacheItem: TypeAlias = tuple[TensorImage, TextInfo]
CacheKey: TypeAlias = str
