
# AWS ML Infra Demo — Distributed Training & Inference Optimization

A compact, **production-style** project that demonstrates:
- **Distributed training** with **DDP** and **FSDP** (PyTorch)
- **Mixed precision** (AMP) + **gradient checkpointing**
- **Profiling** (latency, throughput, memory) with `torch.profiler`
- **Inference optimization**: TorchScript + dynamic quantization
- **Synthetic dataset** for fast, offline runs (no downloads)

This is designed to showcase skills for **Annapurna Labs / AWS Neuron** roles (ML Acceleration, Frameworks, Runtime). It runs on **CPU** or **single/multi‑GPU** (CUDA optional).

---

## Quick Start

### 1) Create env & install
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) (Optional) Enable CUDA
If you have GPUs, install a CUDA-enabled PyTorch per https://pytorch.org/get-started/locally/

### 3) Train: DDP or FSDP (auto-fallback to single process if no GPU)
```bash
# DDP (will use all visible GPUs; falls back to single process if none)
python train_ddp_fsdp.py --mode ddp --epochs 2 --batch-size 256 --amp

# FSDP (requires torch>=2.0)
python train_ddp_fsdp.py --mode fsdp --epochs 2 --batch-size 256 --amp --ckpt

# Single-process baseline
python train_ddp_fsdp.py --mode single --epochs 2 --batch-size 256 --amp
```

### 4) Inference benchmarks
```bash
python benchmark_inference.py --checkpoint artifacts/model.pt --batch-size 1024
```

This will compare **eager FP32**, **AMP**, **TorchScript**, and **dynamic quantization** latencies/throughputs and save a report to `artifacts/inference_report.json`.

### 5) Export artifacts for your application
- Training logs & profiler traces: `artifacts/`
- Final model checkpoint: `artifacts/model.pt`
- Inference report: `artifacts/inference_report.json`

Upload these as evidence of **training scale-up** and **inference optimization**.

---

## Project Layout

```
aws-ml-infra-demo/
  ├─ train_ddp_fsdp.py           # single, DDP, FSDP training (AMP, checkpointing, profiler)
  ├─ benchmark_inference.py      # eager vs AMP vs TorchScript vs quantized timings
  ├─ data/
  │   └─ synthetic.py            # fast synthetic dataset
  ├─ models/
  │   └─ tiny_cnn.py             # tiny ConvNet for speed
  ├─ utils/
  │   ├─ common.py               # helpers (seed, amp scaler, profiler)
  │   └─ dist.py                 # DDP/FSDP setup utilities
  ├─ requirements.txt
  ├─ README.md
  └─ LICENSE
```

---

## Talking Points for Resume / Application

- **Distributed Training:** Implemented DDP & FSDP with AMP and gradient checkpointing; measured **throughput (img/s)** and **peak memory**.
- **Runtime & Profiling:** Used `torch.profiler` to capture operator‑level traces; optimized batch size and autocast policy.
- **Inference Acceleration:** Compared eager FP32 vs **AMP vs TorchScript vs int8 quantization**; achieved **X–Y% latency reduction** on CPU.
- **Production Hygiene:** Deterministic seeding, modular code, CLI args, clear artifacts folder, and reproducible runs.

*(Replace X–Y% with your measured numbers from `artifacts/inference_report.json`.)*

---

## Notes

- FSDP requires `torch>=2.0`. If unavailable, use `--mode ddp` or `--mode single`.
- Dynamic quantization applies to Linear layers (most impactful on CPU).
- TorchScript improves inference portability; for GPU kernels try Triton/TensorRT in extended work.
