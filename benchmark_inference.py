
import argparse, time, json, os, torch
from models.tiny_cnn import TinyCNN
from utils.common import ensure_artifacts

def rand_inputs(batch_size=1024, image_size=(3,32,32), device='cpu'):
    return torch.randn((batch_size, *image_size), device=device)

@torch.no_grad()
def bench(fn, x, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup): fn(x)
    torch.cuda.synchronize() if x.device.type=='cuda' else None
    # Timed
    start = time.time()
    for _ in range(iters): fn(x)
    torch.cuda.synchronize() if x.device.type=='cuda' else None
    elapsed = time.time() - start
    return elapsed/iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='artifacts/model.pt')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    ensure_artifacts()
    device = torch.device(args.device)
    model = TinyCNN().to(device).eval()

    if os.path.exists(args.checkpoint):
        try:
            sd = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print('[WARN] Failed to load checkpoint:', e)

    x = rand_inputs(args.batch_size, device=device)

    # Eager FP32
    t_fp32 = bench(lambda t: model(t.float()), x.float())

    # AMP (CUDA only)
    if device.type == 'cuda':
        def amp_fn(t):
            with torch.cuda.amp.autocast():
                return model(t)
        t_amp = bench(amp_fn, x.float())
    else:
        t_amp = None

    # TorchScript
    try:
        ts = torch.jit.trace(model, x)
        ts.eval()
        t_ts = bench(lambda t: ts(t), x)
    except Exception as e:
        print('[WARN] TorchScript trace failed:', e)
        t_ts = None

    # Dynamic Quantization (CPU only; Linear layers)
    if device.type == 'cpu':
        qmodel = torch.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
        t_int8 = bench(lambda t: qmodel(t), x.cpu())
    else:
        t_int8 = None

    report = {
        'device': str(device),
        'batch_size': args.batch_size,
        'latency_ms': {
            'fp32': round(t_fp32*1000, 3) if t_fp32 else None,
            'amp': round(t_amp*1000, 3) if t_amp else None,
            'torchscript': round(t_ts*1000, 3) if t_ts else None,
            'int8_dynamic': round(t_int8*1000, 3) if t_int8 else None
        },
        'throughput_sps': {
            'fp32': round(args.batch_size / t_fp32, 1) if t_fp32 else None,
            'amp': round(args.batch_size / t_amp, 1) if t_amp else None,
            'torchscript': round(args.batch_size / t_ts, 1) if t_ts else None,
            'int8_dynamic': round(args.batch_size / t_int8, 1) if t_int8 else None
        }
    }
    with open('artifacts/inference_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
