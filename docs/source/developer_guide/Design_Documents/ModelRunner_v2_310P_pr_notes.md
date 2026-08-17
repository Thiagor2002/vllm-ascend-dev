# 310P Model Runner V2 First-Release PR Notes

This document is the community-merge companion for the first 310P Model Runner V2
(MRv2) drop. Use it together with:

- [310P Model Runner V2 adaptation guide](ModelRunner_v2_310P_adaptation.md)
  (scope, acceptance, and staged delivery)
- [310P Model Runner V2 code-change log](ModelRunner_v2_310P_code_changes.md)
  (per-issue root cause and fix history)

Enable the path with `VLLM_USE_V2_MODEL_RUNNER=1`. The default remains Model
Runner V1 (`NPUModelRunner310`).

## 1. What this PR does

310P cannot run the shared Ascend MRv2 path as-is: it has no Triton, Attention
KV cache must be allocated as `ACL_FORMAT_FRACTAL_NZ`, and ACL Graph capture
must record direct NPU operators rather than mainline graph-task handles.

This first release adds a thin 310P MRv2 stack under `vllm_ascend/_310p/worker/v2/`
and a small set of public MRv2 extension points. The product goal is:

> Qwen3-Dense / Qwen3-VL / Qwen3-MoE / Qwen3.5-Dense / Qwen3.5-MoE
> with **TP + ACL Graph (`FULL_DECODE_ONLY`) + W8A8/W8A8SC/W8A8-Dynamic**,
> without prefix cache or MTP. Dense linear layers accept all three schemes;
> 310P grouped-MoE accepts W8A8-Dynamic expert descriptions only.

Prefix cache, full sampling postprocessing, and MTP stay rejected at startup
and are deferred to the second 310P MRv2 release. Closing
`VLLM_USE_V2_MODEL_RUNNER` must leave V1 behavior unchanged.

## 2. Change structure

```text
Public MRv2 (shared, 910B/C + 310P)
  worker/v2/model_runner.py     request_state_cls / aclgraph_manager_cls
                                + overridable _prepare_* helpers
  worker/v2/attn_utils.py       NZ-safe K/V reshape (do not clobber fractal 16)
  worker/v2/model_states/       310P factory branch + hybrid metadata
  patch/platform/patch_use_v2_model_runner.py
                                env-gated V2; skip HAS_TRITON on 310P only
  patch/worker/patch_v2/        310P BlockTables substitution

310P MRv2 (device-local, imported only by the V2 runner)
  _310p/worker_310p.py          V1/V2 runner selection
  _310p/worker/v2/model_runner.py
  _310p/worker/v2/{block_table,states,model_state,rope,aclgraph,sampler,
                   kernel_registry,kv_block_zeroer}.py
  _310p/attention/metadata_builder.py
                                host vs device seq_lens by attention state
  _310p/quantization/           unchanged schemes; clearer MoE error

Tests
  tests/ut/_310p/test_model_runner_v2_310p.py
  tests/ut/_310p/quantization/test_w8a8sc_310.py
  tests/ut/_310p/quantization/test_modelslim_config_310.py
  tests/e2e/pull_request/one_card/_310p/test_model_runner_v2_310p.py
  tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_310p.py
  tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_moe_310p.py
```

Design rules for reviewers:

1. 310P differences stay in `_310p/`. Shared MRv2 only grows class-level
   hooks or overridable methods. Do not scatter `is_310p()` on hot paths.
2. No dependency on unmerged upstream work. Triton paths are replaced by
   substituting classes and module attributes, the mechanisms this repo
   already uses, so the plugin imports cleanly on vLLM main. `kernel_registry`
   only opts into [vLLM #43048](https://github.com/vllm-project/vllm/pull/43048)
   if that dispatcher is present, and is a no-op today.
3. V1 must not import `_310p/worker/v2/`, and 310P must not import
   `worker/v2/block_table.py`, which defines a Triton kernel at import time.
4. Slot mapping, positions, and seq lens are built from CPU mirrors. Device
   tensors are not copied back with `.cpu()` / `.item()` in the request path.
5. Attention KV cache is created with `torch_npu.empty_with_format(...,
   ACL_FORMAT_FRACTAL_NZ)`. `view()` / `reshape()` cannot produce physical NZ.
6. ACL Graph replay uses preallocated device buffers. Pageable H2D must not
   be captured.

## 3. Support matrix (first release)

Status: **verified** = real-weight serve evidence exists; **e2e-ready** = the
test has landed but was not run in this Windows workspace; **code-compatible**
= the path exists and still needs a published checkpoint plus 310P acceptance.

| Model | TP | ACL Graph `FULL_DECODE_ONLY` | Quantization boundary |
| --- | --- | --- | --- |
| Qwen3-8B | verified TP1/TP2 | verified | W8A8 verified TP1 (+ TP2 e2e-ready); W8A8SC verified TP1; W8A8-Dynamic TP1/TP2 e2e-ready |
| Qwen3-VL-2B-Instruct | verified TP1/TP2 | e2e-ready (encoder eager, decode graph) | linear schemes are code-compatible; quantized VL checkpoint pending |
| Qwen3-VL-8B-Instruct | verified TP1/TP2 | e2e-ready (encoder eager, decode graph) | linear schemes are code-compatible; quantized VL checkpoint pending |
| Qwen3-30B-A3B | verified TP2 | e2e-ready | W8A8-Dynamic experts e2e-ready; static W8A8/W8A8SC experts unsupported |
| Qwen3.5-2B | verified TP1/TP2 | verified | Dense linear schemes code-compatible; 310P checkpoint pending |
| Qwen3.5-4B | e2e-ready TP1/TP2 | e2e-ready | Dense linear schemes code-compatible; 310P checkpoint pending |
| Qwen3.5-27B | verified | e2e-ready TP4 | Dense linear schemes code-compatible; 310P checkpoint pending |
| Qwen3.5-35B-A3B | e2e-ready TP4 | e2e-ready TP4 | W8A8-Dynamic experts code-compatible; static W8A8/W8A8SC experts unsupported |

Quantization boundary (layer registry, shared by V1/V2):

| Layer | W8A8 | W8A8SC | W8A8-Dynamic |
| --- | --- | --- | --- |
| Dense linear | supported | supported | supported; fixed int8 activation contract |
| MoE experts | unsupported on 310P grouped operator | unsupported on 310P grouped operator | supported; weights are transposed to `[E,K,N]` before NZ |

W4A8, EP, PP, DP, CP, LoRA, KV transfer, sleep mode, structured output, and
non-greedy sampling are rejected. Expert Parallel is out of 310P V1 scope
and remains rejected here.

## 4. Test inventory

### Unit tests (CPU)

```bash
pytest -sv tests/ut/_310p/test_model_runner_v2_310p.py
pytest -sv tests/ut/_310p/test_block_table_310p.py
pytest -sv tests/ut/_310p/quantization/test_w8a8sc_310.py
pytest -sv tests/ut/_310p/quantization/test_modelslim_config_310.py
pytest -sv tests/ut/worker/test_attn_utils_v2.py
```

Covered contracts: Triton gate skip on 310P only, no Triton or dispatcher
import from the 310P block tables, NumPy slot mapping across cache groups,
first-release config rejects, NZ KV allocation, capture `seq_lens` refresh,
FULL-graph padding, hybrid model-state routing, W8A8SC `tp_rank != 0`
quant-bias zeroing, MoE static-quant error hint.

### E2E (310P hardware)

```bash
# one card: dense/hybrid TP1 + graph; W8A8/W8A8SC/W8A8-Dynamic; VL eager+graph
pytest -sv tests/e2e/pull_request/one_card/_310p/test_model_runner_v2_310p.py

# four cards: TP2 dense/hybrid/VL; TP2 W8A8/W8A8-Dynamic; TP4 Qwen3.5-27B
pytest -sv tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_310p.py

# four cards: Qwen3-30B-A3B TP2 eager/graph; Qwen3.5-35B-A3B TP4 eager/graph
pytest -sv tests/e2e/pull_request/four_card/_310p/test_model_runner_v2_moe_310p.py
```

Existing 310P V1 files under `tests/e2e/pull_request/{one,four}_card/_310p/`
must keep passing with `VLLM_USE_V2_MODEL_RUNNER` unset.

The W8A8-Dynamic cases require access to
`vllm-ascend/Qwen3-8B-W8A8-Dynamic`. Before running the MoE quantized case,
verify that `vllm-ascend/Qwen3-30B-A3B-W8A8` describes expert weights as
`W8A8_DYNAMIC`; static W8A8/W8A8SC expert descriptions are intentionally
rejected on 310P.

### Serve smoke (greedy, no prefix cache)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --cudagraph-mode full_decode_only \
  --no-enable-prefix-caching \
  --max-model-len 8192 --port 8000

# VL, MoE, and Qwen3.5 follow the same flags. Hybrid models also need:
#   --mamba-ssm-cache-dtype float16
# Quantized dense:
#   --quantization ascend
```

Pass criteria for each model: `/v1/models` 200; two consecutive greedy
requests 200 with non-empty output; logs contain
`run 310P full ACL Graph with num_tokens=...`; no Triton compile/invoke.

## 5. User-facing change

Yes, when `VLLM_USE_V2_MODEL_RUNNER=1` on 310P:

- Engine uses `NPUModelRunner310V2`.
- Prefix cache / MTP / non-greedy sampling fail at startup or first request
  with `NotImplementedError`.
- Default (`VLLM_USE_V2_MODEL_RUNNER` unset or `0`) is unchanged V1.

## 6. Out of scope (do not block this PR)

- Prefix cache and MTP (second 310P MRv2 release).
- Temperature / top-k / top-p / penalties / logprobs / grammar.
- Real-weight W8A8-Dynamic accuracy sign-off on 310P hardware. The runtime
  contract and unit/E2E coverage are included; remaining model/checkpoint
  combinations are explicitly marked pending hardware acceptance.
- Static W8A8/W8A8SC MoE schemes (no 310P operator; registry refuses them).
- Qwen3-Embedding / pooling, Gemma4, EP, piecewise ACL Graph as a new 310P
  product mode.

## 7. Reviewer checklist

- [ ] Shared MRv2 changes are extension points only; `worker/v2/block_table.py`
      is untouched, so 910B/C V2 keeps the default Triton slot mapping and
      `ModelAclGraphManager`.
- [ ] Nothing imports `vllm.model_executor.triton_dispatcher` unconditionally.
- [ ] 310P V1 path does not import `_310p/worker/v2/`.
- [ ] First-release guards still reject prefix cache, MTP, EP, PP/DP/CP,
      LoRA, KV transfer, sleep mode.
- [ ] Attention KV is NZ at allocation time; Mamba/GDN state stays ND with
      contiguous per-state stride.
- [ ] `FULL_DECODE_ONLY` capture/replay uses resident device `seq_lens` /
      `query_start_loc` for decode/splitfuse, host `seq_lens` for
      PrefillNoCache.
- [ ] New E2E files live under `*_310p.py` so CI routes them to 310P runners.
- [ ] No new environment variable except the existing
      `VLLM_USE_V2_MODEL_RUNNER`.
- [ ] Commit messages follow Conventional Commits and are signed off.

Suggested PR title:

```text
[Feat][310P] Add Qwen3/Qwen3.5 MRv2 with TP, ACL Graph, and W8A8 variants
```

## 8. Follow-up after merge

1. Second release: prefix cache (block reuse is already implemented;
   startup reject is the remaining product gate).
2. Second release: greedy-plus sampling and MTP speculator under
   `_310p/worker/v2/spec_decode/`.
3. Run the 310P accuracy and throughput acceptance matrix for W8A8-Dynamic
   Dense and MoE checkpoints; keep the fixed contract unless hardware evidence
   requires a reviewed operator change.
4. Add 310P W8A8 checkpoints for Qwen3.5 hybrid once they exist.
5. Reduce `NPUModelRunner310V2.initialize_kv_cache` duplication by hooking
   `_allocate_kv_cache_tensors` in the shared runner.
6. If [vLLM #43048](https://github.com/vllm-project/vllm/pull/43048) merges,
   register the 310P implementations in `kernel_registry.KERNEL_IMPLS` and drop
   the subclass overrides that only exist to bypass a Triton kernel.
