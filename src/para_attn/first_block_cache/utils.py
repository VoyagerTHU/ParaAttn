import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union

import torch

import para_attn.primitives as DP
from .taylorseer import TaylorSeer


@dataclasses.dataclass
class CacheContext:
    residual_diff_threshold: Union[torch.Tensor, float] = 0.0
    alter_residual_diff_threshold: Optional[Union[torch.Tensor, float]] = None

    downsample_factor: int = 1

    enable_alter_cache: bool = False
    num_inference_steps: int = -1
    warmup_steps: int = 0

    enable_taylorseer: bool = False
    taylorseer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    slg_layers: Optional[List[int]] = None
    slg_start: float = 0.0
    slg_end: float = 0.1

    taylorseer: Optional[TaylorSeer] = None
    alter_taylorseer: Optional[TaylorSeer] = None

    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    executed_steps: int = 0
    is_alter_cache: bool = True

    def __post_init__(self):
        if self.enable_taylorseer:
            self.taylorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.enable_alter_cache:
                self.alter_taylorseer = TaylorSeer(**self.taylorseer_kwargs)

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    def get_residual_diff_threshold(self):
        if self.enable_alter_cache and self.is_alter_cache and self.alter_residual_diff_threshold is not None:
            residual_diff_threshold = self.alter_residual_diff_threshold
        else:
            residual_diff_threshold = self.residual_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        return residual_diff_threshold

    def get_buffer(self, name):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        self.buffers[name] = buffer

    def remove_buffer(self, name):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        if name in self.buffers:
            del self.buffers[name]

    def clear_buffers(self):
        self.buffers.clear()

    def mark_step_begin(self):
        if not self.enable_alter_cache:
            self.executed_steps += 1
        else:
            self.is_alter_cache = not self.is_alter_cache
            if not self.is_alter_cache:
                self.executed_steps += 1
        if self.enable_taylorseer:
            taylorseer = self.get_taylorseer()
            taylorseer.mark_step_begin()

    def get_taylorseer(self):
        if self.enable_alter_cache and self.is_alter_cache:
            return self.alter_taylorseer
        return self.taylorseer

    def is_slg_enabled(self):
        return self.slg_layers is not None

    def slg_should_skip_block(self, block_idx):
        if not self.enable_alter_cache or not self.is_alter_cache:
            return False
        if self.slg_layers is None:
            return False
        if self.slg_start <= 0.0 and self.slg_end >= 1.0:
            return False
        num_inference_steps = self.num_inference_steps
        assert num_inference_steps >= 0, "num_inference_steps must be non-negative"
        return (
            block_idx in self.slg_layers
            and num_inference_steps * self.slg_start <= self.get_current_step() < num_inference_steps * self.slg_end
        )

    def get_current_step(self):
        return self.executed_steps - 1

    def is_in_warmup(self):
        return self.get_current_step() < self.warmup_steps


@torch.compiler.disable
def get_residual_diff_threshold():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_residual_diff_threshold()


@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


@torch.compiler.disable
def remove_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.remove_buffer(name)


@torch.compiler.disable
def mark_step_begin():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.mark_step_begin()


@torch.compiler.disable
def is_taylorseer_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_taylorseer


@torch.compiler.disable
def get_taylorseer():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_taylorseer()


@torch.compiler.disable
def is_slg_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_slg_enabled()


@torch.compiler.disable
def slg_should_skip_block(block_idx):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.slg_should_skip_block(block_idx)


@torch.compiler.disable
def is_in_warmup():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_in_warmup()


_current_cache_context = None


def create_cache_context(*args, **kwargs):
    return CacheContext(*args, **kwargs)


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    if threshold <= 0.0:
        return False

    if t1.shape != t2.shape:
        return False

    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    if parallelized:
        mean_diff = DP.all_reduce_sync(mean_diff, "avg")
        mean_t1 = DP.all_reduce_sync(mean_t1, "avg")
    diff = mean_diff / mean_t1
    return diff.item() < threshold


@torch.compiler.disable
def apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states=None):
    if is_taylorseer_enabled():
        hidden_states_residual = get_hidden_states_residual()
        assert hidden_states_residual is not None, "hidden_states_residual must be set before"
        hidden_states = hidden_states_residual + hidden_states

        hidden_states = hidden_states.contiguous()
    else:
        hidden_states_residual = get_hidden_states_residual()
        assert hidden_states_residual is not None, "hidden_states_residual must be set before"
        hidden_states = hidden_states_residual + hidden_states

        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            encoder_hidden_states_residual = get_encoder_hidden_states_residual()
            assert encoder_hidden_states_residual is not None, "encoder_hidden_states_residual must be set before"
            encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states

            encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable
def get_downsample_factor():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.downsample_factor


@torch.compiler.disable
def get_can_use_cache(first_hidden_states_residual, parallelized=False):
    if is_in_warmup():
        return False
    threshold = get_residual_diff_threshold()
    if threshold <= 0.0:
        return False
    downsample_factor = get_downsample_factor()
    if downsample_factor > 1:
        first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]
    prev_first_hidden_states_residual = get_first_hidden_states_residual()
    can_use_cache = prev_first_hidden_states_residual is not None and are_two_tensors_similar(
        prev_first_hidden_states_residual,
        first_hidden_states_residual,
        threshold=threshold,
        parallelized=parallelized,
    )
    return can_use_cache


@torch.compiler.disable
def set_first_hidden_states_residual(first_hidden_states_residual):
    downsample_factor = get_downsample_factor()
    if downsample_factor > 1:
        first_hidden_states_residual = first_hidden_states_residual[..., ::downsample_factor]
        first_hidden_states_residual = first_hidden_states_residual.contiguous()
    set_buffer("first_hidden_states_residual", first_hidden_states_residual)


@torch.compiler.disable
def get_first_hidden_states_residual():
    return get_buffer("first_hidden_states_residual")


@torch.compiler.disable
def set_hidden_states_residual(hidden_states_residual):
    if is_taylorseer_enabled():
        taylorseer = get_taylorseer()
        taylorseer.update(hidden_states_residual)
    else:
        set_buffer("hidden_states_residual", hidden_states_residual)


@torch.compiler.disable
def get_hidden_states_residual():
    if is_taylorseer_enabled():
        taylorseer = get_taylorseer()
        return taylorseer.approximate_value()
    else:
        return get_buffer("hidden_states_residual")


@torch.compiler.disable
def set_encoder_hidden_states_residual(encoder_hidden_states_residual):
    if is_taylorseer_enabled():
        return
    set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)


@torch.compiler.disable
def get_encoder_hidden_states_residual():
    return get_buffer("encoder_hidden_states_residual")


class CachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        return_hidden_states_first=True,
        return_hidden_states_only=False,
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states, encoder_hidden_states = hidden_states
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        mark_step_begin()
        can_use_cache = get_can_use_cache(
            first_hidden_states_residual,
            parallelized=self.transformer is not None and getattr(self.transformer, "_is_parallelized", False),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                hidden_states, encoder_hidden_states
            )
        else:
            set_first_hidden_states_residual(first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_remaining_transformer_blocks(hidden_states, encoder_hidden_states, *args, **kwargs)
            set_hidden_states_residual(hidden_states_residual)
            set_encoder_hidden_states_residual(encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        return (
            hidden_states
            if self.return_hidden_states_only
            else (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )
        )

    def call_remaining_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        if not is_slg_enabled():
            for block in self.transformer_blocks[1:]:
                hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )
        else:
            for i, encoder_block in enumerate(self.transformer_blocks[1:]):
                if slg_should_skip_block(i + 1):
                    continue
                hidden_states = encoder_block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for i, block in enumerate(self.single_transformer_blocks):
                    if slg_should_skip_block(len(self.transformer_blocks) + i):
                        continue
                    hidden_states = block(hidden_states, *args, **kwargs)
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )

        # hidden_states_shape = hidden_states.shape
        # encoder_hidden_states_shape = encoder_hidden_states.shape
        hidden_states = hidden_states.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        # hidden_states = hidden_states.contiguous()
        # encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states

        hidden_states_residual = hidden_states_residual.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states_residual = (
            encoder_hidden_states_residual.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual
