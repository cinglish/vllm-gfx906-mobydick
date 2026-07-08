# MiniMax M3 on gfx906 (MI60 8x 32GB)

## Patch model config (one-time, run after model downloads)

The VL checkpoint needs config patches to load as text-only via ForCausalLM:
- Remove `auto_map` so vllm uses its built-in MiniMaxM3Config
- Set architecture to `MiniMaxM3SparseForCausalLM` (skips vision tower)
- Keep `quantization_config` as-is (auto-round routes through INC to GPTQ/WNA16
  backends; handles per-layer FP16 overrides for gates, embeddings, vision)
- Fix `tokenizer_class` for transformers 4.56 compat

Requires `inc` in ROCm's `supported_quantization` list (patched in vllm fork).

## Hotpatch vllm (if not rebuilt from fork)

Apply these if running a stock vllm image instead of building from the fork.

**From outside the container** (kubectl):
```bash
# Fix sparse attention D_CHUNK slice indexing for older Triton
kubectl cp vllm-gfx906/vllm/models/minimax_m3/common/ops/sparse_attn.py \
  $POD:/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/sparse_attn.py
```

**Inside the container** (kubectl exec or shell):
```bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

# Allow INC/auto-round quantization on ROCm
sed -i '/"bitsandbytes",/a\        "inc",  # routes auto-round to GPTQ/WNA16 backends' \
  $VLLM/platforms/rocm.py

# Fix INC returning UnquantizedLinearMethod for embedding layers
sed -i 's/                    return UnquantizedLinearMethod()/                    if isinstance(layer, (LinearBase, ParallelLMHead)):\n                        return UnquantizedLinearMethod()\n                    return None/' \
  $VLLM/model_executor/layers/quantization/inc.py

# Clear Triton cache so kernels recompile
rm -rf /models/.cache/vllm/triton /root/.triton/cache
```

```bash
python3 << 'PATCH'
import json, os
from huggingface_hub import snapshot_download

path = snapshot_download('bullerwins/MiniMax-M3-4bit-W4A16-v0')
cfg = os.path.join(path, 'config.json')

# Remove both the symlink AND the underlying blob to force a fresh download
# (huggingface_hub caches blobs separately from symlinks)
real = os.path.realpath(cfg)
os.remove(cfg)
if os.path.exists(real):
    os.remove(real)
path = snapshot_download('bullerwins/MiniMax-M3-4bit-W4A16-v0')

with open(cfg) as f: d = json.load(f)
d.pop('auto_map', None)
d['architectures'] = ['MiniMaxM3SparseForCausalLM']
# Strip language_model. prefix from extra_config keys so they match
# vllm's internal layer names (WeightsMapper strips this prefix)
qc = d.get('quantization_config', {})
ec = qc.get('extra_config', {})
if ec:
    qc['extra_config'] = {
        k.removeprefix('language_model.'): v for k, v in ec.items()
    }
with open(cfg, 'w') as f: json.dump(d, f, indent=2)
print('patched config.json')
print('quant_method:', d.get('quantization_config', {}).get('quant_method'))

tc = os.path.join(path, 'tokenizer_config.json')
with open(tc) as f: d = json.load(f)
d['tokenizer_class'] = 'PreTrainedTokenizerFast'
with open(tc, 'w') as f: json.dump(d, f, indent=2)
print('patched tokenizer_config.json')
print('cache path:', path)
PATCH
```

## Serve command (breakable cudagraph)

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=1 python3 -m vllm.entrypoints.openai.api_server --model bullerwins/MiniMax-M3-4bit-W4A16-v0 --tensor-parallel-size 4 --pipeline-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 4096 --dtype half --trust-remote-code --kv-cache-dtype fp8 --language-model-only --distributed-timeout-seconds 1800
```

## Serve command (eager fallback)

```bash
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=4 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2 python3 -m vllm.entrypoints.openai.api_server --model bullerwins/MiniMax-M3-4bit-W4A16-v0 --tensor-parallel-size 4 --pipeline-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 8 --dtype half --trust-remote-code --kv-cache-dtype fp8 --enforce-eager --language-model-only --distributed-timeout-seconds 1800 --disable-custom-all-reduce
```

## Profiling with rocprof

Eager mode is required — HIP graphs collapse all kernel launches into a single
`hipGraphLaunch` event, hiding per-kernel timing.

**Step 1: Warm up Triton caches** (normal server, no profiler). Start the eager
fallback server above, send a few requests to trigger JIT compilation, then
Ctrl+C. Caches persist at `/models/.cache/vllm`.

**Step 2: Restart under rocprof** with warm caches for a clean trace.

```bash
mkdir -p /tmp/m3-profile
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 HSA_ENABLE_SDMA=0 \
  HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=4 \
  NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2 \
  rocprof --hip-trace --hsa-trace --roctx-trace \
  -o /tmp/m3-profile/results.csv \
  python3 -m vllm.entrypoints.openai.api_server \
  --model bullerwins/MiniMax-M3-4bit-W4A16-v0 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --max-num-seqs 8 --dtype half --trust-remote-code \
  --kv-cache-dtype fp8 --enforce-eager \
  --language-model-only --distributed-timeout-seconds 1800 --disable-custom-all-reduce
```

Send test requests, then Ctrl+C. Results in `/tmp/m3-profile/`.

## Notes

- First request triggers Triton JIT compilation (~15-30 min). Cached at /models/.cache/vllm after.
- `--distributed-timeout-seconds 1800` sets NCCL timeout to 30 min (default 600s too short for JIT)
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` sets vLLM's RPC timeout to match
- 227GB model in 256GB VRAM (~5.5GB vision weights skipped = ~34GB headroom)
- TP=4 max due to 4 KV heads; PP=2 to use all 8 GPUs
- `num_warps=8` override on gfx906 reduces register spill in sparse attention kernels
- `HSA_ENABLE_SDMA=0` fixes intermittent GPU memory access faults
- FP8 KV cache halves KV memory footprint
- `NCCL_*CHANNELS` reduces NCCL overhead for small PP transfers over PCIe
