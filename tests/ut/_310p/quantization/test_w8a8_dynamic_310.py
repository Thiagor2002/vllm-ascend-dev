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

from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    _MIN_NZ_QUANT_MATMUL_N,
    AscendW8A8DynamicFusedMoEMethod310,
    AscendW8A8DynamicLinearMethod310,
)


class TestAscendW8A8FusedMoEMethod310(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_ep_group")
    def setUp(self, mock_get_ep_group):
        with patch(
            "vllm_ascend._310p.quantization.methods.w8a8_dynamic.get_current_vllm_config"
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.quant_config = Mock(quant_description={"group_size": 0})
            mock_vllm_config.scheduler_config = Mock(
                max_num_batched_tokens=2048, max_model_len=2048, enable_chunked_prefill=False
            )
            mock_get_current_vllm_config.return_value = mock_vllm_config
            mock_ep_group = Mock()
            mock_get_ep_group.return_value = mock_ep_group
            mock_ascend_config = Mock()

            mock_ascend_config.enable_chunked_prefill = False

            self.quant_method = AscendW8A8DynamicFusedMoEMethod310()

    def test_get_weight_310(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(
            param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.assertEqual(param_dict["w2_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size))

    def test_get_dynamic_quant_param_310(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.float16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (self.num_experts, 2 * self.intermediate_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (self.num_experts, self.hidden_size, 1))


class TestAscendW8A8DynamicLinearMethod310(TestBase):
    def setUp(self):
        self.method = AscendW8A8DynamicLinearMethod310()

    def test_get_weight_310(self):
        weight = self.method.get_weight(10, 20)
        self.assertEqual(weight["weight"].dtype, torch.int8)
        self.assertEqual(weight["weight"].shape, (20, 10))

    def test_get_perchannel_param_310(self):
        params = self.method.get_perchannel_param(10, torch.float32)

        self.assertEqual(params["weight_scale"].dtype, torch.float32)
        self.assertEqual(params["weight_offset"].dtype, torch.float32)

        self.assertEqual(params["weight_scale"].shape, (10, 1))
        self.assertEqual(params["weight_offset"].shape, (10, 1))

    def test_apply_310(self):
        layer = MagicMock()
        weight = torch.randint(-8, 8, (256, 128), dtype=torch.int8)
        scale = torch.ones(256, dtype=torch.float32)
        layer.weight_fp = (weight.to(torch.float16) * scale.view(-1, 1)).to(torch.float16)
        x = torch.randn(32, 128, dtype=torch.float16)

        output = self.method.apply(layer, x, tp_rank=0)

        self.assertEqual(output.shape, (32, 256))
        self.assertEqual(output.dtype, torch.float16)

    def test_apply_fp16_fallback_skips_quant_matmul_310(self):
        # Document the N=256 case that triggers QuantBatchMatmulV3_NZ_NZ kernel 21
        # under GE when NZ weights are used (Qwen3.5-2B TP2 KV shard).
        self.assertEqual(_MIN_NZ_QUANT_MATMUL_N, 512)
        layer = MagicMock()
        weight = torch.randint(-8, 8, (256, 2048), dtype=torch.int8)
        scale = torch.ones(256, dtype=torch.float32)
        layer.weight_fp = (weight.to(torch.float16) * scale.view(-1, 1)).to(torch.float16)
        x = torch.randn(4, 2048, dtype=torch.float16)

        output = self.method.apply(layer, x, tp_rank=0)

        self.assertEqual(output.shape, (4, 256))
        self.assertEqual(output.dtype, torch.float16)

    def test_process_weights_keeps_nd_and_builds_weight_fp(self):
        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight_scale = MagicMock()
        layer.weight_offset = MagicMock()
        layer.params_dtype = torch.float16

        layer.weight.data = torch.randint(-8, 8, (256, 128), dtype=torch.int8)
        layer.weight_scale.data = torch.ones(256, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.zeros(256, 1, dtype=torch.float32)

        self.method.process_weights_after_loading(layer)

        self.assertTrue(hasattr(layer, "weight_fp"))
        self.assertEqual(layer.weight_fp.shape, (256, 128))
        self.assertEqual(layer.weight_fp.dtype, torch.float16)
        self.assertEqual(layer.weight_scale.data.ndim, 1)
