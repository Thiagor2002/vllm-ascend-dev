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

from typing import Any, cast

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType
from vllm_ascend.utils import maybe_trans_nz

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod310(AscendMoEScheme):
    """310P-only FusedMoE method for Ascend W8A8_DYNAMIC.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8
    # Activation quant dtype used by the MLP gmm hooks.
    act_quant_type: torch.dtype = torch.int8
    # 310P gmm1+swiglu+quant is fused inside npu_quant_grouped_matmul_dequant
    # + npu_swiglu, so silu is handled by ``apply_gmm1_act_quant``.
    fused_activations = frozenset({"silu"})

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
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=getattr(layer, "activation", "silu"),
            ),
            quant_method=self,
        )
        return final_hidden_states

    def _get_group_list(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        """Return the cumulative-sum group_list expected by 310P kernels."""
        group_list = mlp_compute_input.group_list
        if mlp_compute_input.group_list_type == 1:
            # Convert group_list to cumulative sum format if group_list is count format
            group_list = torch.cumsum(group_list, dim=0)
        return group_list

    def _get_mlp_weights(self, layer: torch.nn.Module) -> tuple:
        """Return (w1, w1_scale, w2, w2_scale) in the standard MLP layout."""
        return (
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
        )

    def get_mlp_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Standard MLP-layout weights (w1/w2 with their scales)."""
        w1, w1_scale, w2, w2_scale = self._get_mlp_weights(layer)
        return MoEWeights(
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        """Fused gmm1 (quant + dequant) + swiglu via 310P kernels.

        ``npu_quant_grouped_matmul_dequant`` quantizes the hidden states to
        int8 internally (``quant_mode="pertoken"``) and dequantizes the gmm1
        output, so the activation scale is not needed by ``apply_gmm2``.
        """
        layer = mlp_compute_input.layer
        assert layer is not None
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
            x=mlp_compute_input.hidden_states,
            quantized_weight=w1,
            weight_scale=w1_scale,
            group_list=self._get_group_list(mlp_compute_input),
            quant_mode="pertoken",
        )
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        return hidden_states, None

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        """down projection (gmm2, quant + dequant)."""
        layer = mlp_compute_input.layer
        assert layer is not None
        _, _, w2, w2_scale = self._get_mlp_weights(layer)
        return torch_npu.npu_quant_grouped_matmul_dequant(
            x=hidden_states,
            quantized_weight=w2,
            weight_scale=w2_scale,
            group_list=self._get_group_list(mlp_compute_input),
            quant_mode="pertoken",
        )

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)


def _is_qwen35_2b_hidden() -> bool:
    """Qwen3.5-2B is uniquely sensitive on 310P W8A8-Dynamic (#14335)."""
    try:
        model_config = get_current_vllm_config().model_config
        text_config = getattr(model_config, "hf_text_config", None)
        hidden = getattr(text_config, "hidden_size", None) if text_config is not None else None
        if hidden is None:
            hidden = getattr(model_config, "hidden_size", None)
        return hidden is not None and int(hidden) == 2048
    except Exception:
        return False


@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 dynamic linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
      - Default: ``npu_dynamic_quant`` + ``npu_quant_matmul`` on NZ ``[K, N]``
        (``nz_then_t``) — true W8A8-Dynamic with int8 NZ GEMM.
      - Qwen3.5-2B (hidden=2048): fused ``npu_quant_matmul_dequant`` with ND
        ``[N, K]`` int8 weights (still pertoken dynamic act quant + int8 GEMM;
        avoids NZ split-path error accumulation on the small 2B hybrid).
      - Do **not** use load-time fp16 dequant / ``F.linear`` (pseudo-quant).
    """

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
        del tp_rank
        # Flatten ND→2D: concurrent MTP/FULL paths may pass [..., K].
        original_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, original_shape[-1])
        x = x.contiguous()

        if getattr(layer, "_310p_w8a8_fused_dequant", False):
            # Qwen3.5-2B: fused pertoken dynamic quant + int8 matmul (ND weight).
            output = torch_npu.npu_quant_matmul_dequant(
                x,
                layer.weight.data,
                layer.weight_scale,
                bias=bias,
                quant_mode="pertoken",
            )
        else:
            # Default 310P path: split dynamic quant + NZ quant_matmul.
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
            if pertoken_scale.dim() > 1:
                quantized_x = quantized_x.reshape(-1, quantized_x.shape[-1])
                pertoken_scale = pertoken_scale.reshape(-1)
            output = torch_npu.npu_quant_matmul(
                quantized_x,
                layer.weight.data,
                layer.weight_scale,
                pertoken_scale=pertoken_scale,
                bias=bias,
                output_dtype=x.dtype,
            )
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], output.shape[-1])
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = cast(torch.Tensor, layer.weight.data)
        weight_scale = cast(torch.Tensor, layer.weight_scale.data)
        weight_offset = cast(torch.Tensor, layer.weight_offset.data)
        weight_scale = weight_scale.flatten()
        weight_offset = weight_offset.flatten()
        layer.weight_scale.data = weight_scale
        layer.weight_offset.data = weight_offset
        # 2B: keep ND [N, K] for npu_quant_matmul_dequant; others: NZ [K, N].
        if _is_qwen35_2b_hidden():
            layer._310p_w8a8_fused_dequant = True
            layer.weight.data = weight.contiguous()
        else:
            layer._310p_w8a8_fused_dequant = False
            layer.weight.data = maybe_trans_nz(weight).transpose(0, 1)
