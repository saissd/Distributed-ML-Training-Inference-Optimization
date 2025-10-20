
import torch, time, os, json, contextlib

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@contextlib.contextmanager
def maybe_autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield

def ensure_artifacts():
    os.makedirs("artifacts", exist_ok=True)

def save_metrics(path, metrics: dict):
    ensure_artifacts()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def measure_throughput(start_time, processed_items):
    elapsed = time.time() - start_time
    ips = processed_items / max(elapsed, 1e-8)
    return elapsed, ips
