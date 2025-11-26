from core.config import *
from core.registry import Registry
from core.viz import get_layer_numel_str

import torch
from torch import nn

from core._types import ABC, Optional
from functools import partial

class BottleneckAdapter(nn.Module, ABC):
    """
    TODO
    """
    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)


BOTTLENECK_ADAPTERS_REGISTRY = Registry[BottleneckAdapter]()


class PoolerLinear(BottleneckAdapter):
    """
    TODO
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            pooler: nn.Module,
            device: torch.device,
    ) -> None:
        super().__init__()

        self.pooler = pooler
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(in_features, out_features, True, device=device)
        
        self._init_weights()

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        # inputs: (B, C_in, H, W)
        x: torch.Tensor = self.pooler(inputs) # (B, C_in, 1, 1)
        x = self.flatten(x) # (B, C_in)
        x = self.linear(x) # (B, C_out)
        return x
    
    def _init_weights(
            self,
    ) -> None:
        # linear: Xavier (no ReLU)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)


class PoolerLinear2BN(BottleneckAdapter):
    """
    TODO
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            pooler: nn.Module,
            device: torch.device,
            mid_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_features is None:
            mid_features = out_features

        self.pooler = pooler
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features, mid_features, False, device=device)
        self.bn1 = nn.BatchNorm1d(mid_features, device=device)
        self.relu1 = nn.ReLU(inplace=True) # inplace saves memory

        self.linear2 = nn.Linear(mid_features, out_features, True, device=device)
        
        self._init_weights()

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        # inputs: (B, C_in, H, W)
        x: torch.Tensor = self.pooler(inputs) # (B, C_in, 1, 1)
        x = self.flatten(x) # (B, c_in)
        x = self.linear1(x) # (B, C_mid)
        x = self.bn1(x) # (B, C_mid)
        x = self.relu1(x) # (B, C_mid)
        x = self.linear2(x) # (B, C_out)
        return x

    def _init_weights(
            self,
    ) -> None:
        # linear1: Kaiming for ReLU
        nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        # linear2: Xavier (no ReLU)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

        # batch norm 1
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)


class PoolerLinear2NoBN(BottleneckAdapter):
    """
    TODO
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            pooler: nn.Module,
            device: torch.device,
            mid_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_features is None:
            mid_features = out_features

        self.pooler = pooler
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features, mid_features, True, device=device)
        self.relu1 = nn.ReLU(inplace=True) # inplace saves memory

        self.linear2 = nn.Linear(mid_features, out_features, True, device=device)
        
        self._init_weights()

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        # inputs: (B, C_in, H, W)
        x: torch.Tensor = self.pooler(inputs) # (B, C_in, 1, 1)
        x = self.flatten(x) # (B, c_in)
        x = self.linear1(x) # (B, C_mid)
        x = self.relu1(x) # (B, C_mid)
        x = self.linear2(x) # (B, C_out)
        return x

    def _init_weights(
            self,
    ) -> None:
        # linear1: Kaiming for ReLU
        nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        # linear2: Xavier (no ReLU)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)


class Conv2BNPooler(BottleneckAdapter):
    """
    TODO
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            pooler: nn.Module,
            device: torch.device,
            mid_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_features is None:
            mid_features = out_features

        self.conv1 = nn.Conv2d(in_features, mid_features, kernel_size=(1, 1), stride=(1, 1), bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(mid_features, device=device)
        self.relu1 = nn.ReLU(inplace=True) # inplace saves memory

        self.conv2 = nn.Conv2d(mid_features, out_features, kernel_size=(1, 1), stride=(1, 1), bias=True, device=device)

        self.pooler = pooler
        self.flatten = nn.Flatten()
        
        self._init_weights()

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        # inputs: (B, C_in, H, W)
        x: torch.Tensor = self.conv1(inputs) # (B, C_mid, H, W)
        x = self.bn1(x) # (B, C_mid, H, W)
        x = self.relu1(x) # (B, C_mid, H, W)
        x = self.conv2(x) # (B, C_out, H, W)
        x = self.pooler(x) # (B, C_out, 1, 1)
        x = self.flatten(x) # (B, C_out)
        return x

    def _init_weights(
            self,
    ) -> None:
        # conv1: Kaiming for ReLU
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        # conv2: Xavier (no ReLU)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # batch norm 1
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

# adapters with different poolers are registered as separate objects
BOTTLENECK_ADAPTERS_REGISTRY.add('GAP_linear', partial(PoolerLinear, pooler=nn.AdaptiveAvgPool2d(output_size=(1, 1))))
BOTTLENECK_ADAPTERS_REGISTRY.add('GAP_linear2_BN', partial(PoolerLinear2BN, pooler=nn.AdaptiveAvgPool2d(output_size=(1, 1))))
BOTTLENECK_ADAPTERS_REGISTRY.add('GAP_linear2_noBN', partial(PoolerLinear2NoBN, pooler=nn.AdaptiveAvgPool2d(output_size=(1, 1))))
BOTTLENECK_ADAPTERS_REGISTRY.add('conv2_BN_GAP', partial(Conv2BNPooler, pooler=nn.AdaptiveAvgPool2d(output_size=(1, 1))))

def print_gap_linear_params_count() -> None:
    gap_linear = BOTTLENECK_ADAPTERS_REGISTRY.get('GAP_linear', in_features=960, out_features=512, device=torch.device('cpu'))
    print("GAP_linear")
    print(get_layer_numel_str(gap_linear, False, False))
    x = torch.rand(2, 960, 33, 33)
    y = gap_linear(x)
    assert y.shape == (2, 512), f"'GAP_linear(x)' should have shape (2, 512), got {y.shape}"
    print(y.shape)
    print("---")

def print_gap_linear2_bn_params_count() -> None:
    gap_linear2_bn = BOTTLENECK_ADAPTERS_REGISTRY.get('GAP_linear2_BN', in_features=960, out_features=512, device=torch.device('cpu'))
    print("GAP_linear2_BN")
    print(get_layer_numel_str(gap_linear2_bn, False, False))
    x = torch.rand(2, 960, 33, 33)
    y = gap_linear2_bn(x)
    assert y.shape == (2, 512), f"'GAP_linear2_BN(x)' should have shape (2, 512), got {y.shape}"
    print(y.shape)
    print("---")

def print_gap_linear2_nobn_params_count() -> None:
    gap_linear2_nobn = BOTTLENECK_ADAPTERS_REGISTRY.get('GAP_linear2_noBN', in_features=960, out_features=512, device=torch.device('cpu'))
    print("GAP_linear2_noBN")
    print(get_layer_numel_str(gap_linear2_nobn, False, False))
    x = torch.rand(2, 960, 33, 33)
    y = gap_linear2_nobn(x)
    assert y.shape == (2, 512), f"'GAP_linear2_noBN(x)' should have shape (2, 512), got {y.shape}"
    print(y.shape)
    print("---")

def print_conv2_bn_gap_params_count() -> None:
    conv2_bn_gap = BOTTLENECK_ADAPTERS_REGISTRY.get('conv2_BN_GAP', in_features=960, out_features=512, device=torch.device('cpu'))
    print("conv2_BN_GAP")
    print(get_layer_numel_str(conv2_bn_gap, False, False))
    x = torch.rand(2, 960, 33, 33)
    y = conv2_bn_gap(x)
    assert y.shape == (2, 512), f"'conv2_BN_GAP(x)' should have shape (2, 512), got {y.shape}"
    print(y.shape)
    print("---")

if __name__ == '__main__':
    print_gap_linear_params_count()
    print_gap_linear2_bn_params_count()
    print_gap_linear2_nobn_params_count()
    print_conv2_bn_gap_params_count()
