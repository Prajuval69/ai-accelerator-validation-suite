# Synthetic ONNX fixtures & backend selection

## Required workloads
- `mlp_gemm`: Linear(128,256)->ReLU->Linear(256,64), input `[B,128]` float32.
- `conv2d`: 3x3 convolution pipeline + ReLU + global average pool, input `[B,3,64,64]` float32.
- `attention_matmul`: small Q/K/V `matmul` -> softmax -> matmul pipeline, input `[B,16,32]` float32.
- Generate exact seeded weights once per run. Prefer batch sizes 1,4,8; bound memory/time. No training and no remote model downloads.
- Export using available `torch.onnx` APIs with `dynamo=True` where supported; `onnx.checker.check_model` required.

## Providers
- Required `CPUExecutionProvider`.
- Optional `CUDAExecutionProvider` for compatible NVIDIA stack.
- Optional `MIGraphXExecutionProvider` for AMD with appropriate build/driver. ROCm is the ecosystem, **not** the name of the current ORT EP. Do not use removed legacy `ROCMExecutionProvider`.
- `aivs doctor` must show provider availability and requirements but must not install drivers or claim a GPU based only on wheel presence.
- For every workload/backend save requested vs actual provider(s) and confidence in accelerator placement.

## Precision
- P0 float32 validation.
- P1 optional dynamic INT8 quantization on suitable operators; compare output against FP32 via meaningful numeric tolerances and report unsupported models/EPs; never promise a speedup.
- Do not claim FP16 unless actually implemented and verified with a supported provider.

## Correctness
- Before timing, run input through PyTorch reference + requested ORT session and check `atol=1e-4`, `rtol=1e-3` default float32 (configurable and documented).
- For INT8 use a distinct documented tolerance/summary; don't force FP32 tolerances onto quantization blindly.
- Save verdict and mismatch reason; optionally block benchmarking a workload failing correctness (configurable, default block).
