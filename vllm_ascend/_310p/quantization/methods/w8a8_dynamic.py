#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group

from vllm_ascend._310p.fused_moe.experts_selector import select_experts
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType
from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme

# 310P GE retile of FRACTAL_NZ W8A8-Dynamic weights during torch.compile
# launches QuantBatchMatmulV3_NZ_NZ kernel 21 (hash 5247287448945562503).
# Eager NZ works; compiled Qwen3.5-2B TP2 does not (fused qkv KV shard N=256
# and MLP). Linear layers keep ND [N, K] and dequant to fp16. MoE experts
# keep ND [E, N, K] for npu_quant_grouped_matmul_dequant (WeightNZ 3D
# parameters lose FRACTAL_NZ under GE and fail tiling).
_MIN_NZ_QUANT_MATMUL_N = 512


def _needs_fp16_quant_matmul_fallback(_layer: torch.nn.Module, _weight: torch.Tensor) -> bool:
    """310P W8A8-Dynamic linear always uses ND + fp16 dequant.

    GE retile of FRACTAL_NZ weights during torch.compile launches
    ``QuantBatchMatmulV3_NZ_NZ`` kernel 21 (hash 5247287448945562503). Eager NZ
    works; compiled 2B TP2 does not, including fused qkv and MLP. Keep this
    helper so tests can assert the fallback policy.
    """
    return True


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod310(AscendMoEScheme):
    """310P-only FusedMoE method for Ascend W8A8_DYNAMIC.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8

    def __init__(self):
        self.ep_group = get_ep_group()
        vllm_config = get_current_vllm_config()
        self.in_dtype = vllm_config.model_config.dtype

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        # Fused gate_up_proj (column parallel)
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        # down_proj (row parallel)
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=num_experts,
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            ),
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states

    def process_weights_after_loading(self, layer):
        # Keep ModelSlim [E, N, K] as ND. FRACTAL_NZ is applied inside the
        # opaque grouped-matmul custom op so torch.compile/GE cannot strip
        # format-29 from 3D Parameters (Qwen3-30B-A3B-W8A8 MRv2).
        layer.w13_weight.data = layer.w13_weight.data.contiguous()
        layer.w2_weight.data = layer.w2_weight.data.contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)


@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 dynamic linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    act_quant_type: torch.dtype = torch.int8

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # Always ND [N, K] + fp16 dequant. torch.compile/GE cannot safely run
        # 310P QuantBatchMatmulV3_NZ_NZ on these dynamic-quant linears.
        scale = layer.weight_scale.data if hasattr(layer.weight_scale, "data") else layer.weight_scale
        weight_fp = layer.weight.data.to(x.dtype) * scale.to(x.dtype).view(-1, 1)
        bias_term = bias if (tp_rank is None or tp_rank == 0) else None
        return torch.nn.functional.linear(x, weight_fp, bias_term)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.contiguous()
        layer._310p_w8a8_dynamic_fp16_fallback = _needs_fp16_quant_matmul_fallback(layer, layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
