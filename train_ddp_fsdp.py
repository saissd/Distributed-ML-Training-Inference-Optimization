
import os, argparse, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from data.synthetic import SyntheticImageDataset
from models.tiny_cnn import TinyCNN
from utils.common import set_seed, maybe_autocast, ensure_artifacts, save_metrics, measure_throughput
from utils import dist as dist_utils

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type=='cuda') else torch.amp.GradScaler('cpu', enabled=False)
    total_loss, total_correct, total_seen = 0.0, 0, 0
    start = time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(use_amp, device):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += x.size(0)
    elapsed, ips = measure_throughput(start, total_seen)
    return total_loss/total_seen, total_correct/total_seen, ips

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['single','ddp','fsdp'], default='single')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--ckpt', action='store_true', help='enable gradient checkpointing (fsdp-friendly)')
    p.add_argument('--num-samples', type=int, default=50000)
    args = p.parse_args()

    set_seed(42); ensure_artifacts()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SyntheticImageDataset(num_samples=args.num_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))

    model = TinyCNN()
    if args.ckpt:
        # simple example: enable checkpointing by wrapping certain layers if needed
        for m in model.modules():
            if isinstance(m, torch.nn.Sequential):
                m.forward = torch.utils.checkpoint.checkpoint_sequential(m, len(m))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # DDP minimal fallback: use single process if no multi-GPU available
    if args.mode == 'ddp' and world_size <= 1:
        print('[WARN] DDP requested but only one or zero GPUs available — running single process.')
        args.mode = 'single'

    if args.mode == 'fsdp':
        try:
            from torch.distributed import init_process_group, destroy_process_group
            init_process_group(backend='nccl' if device.type=='cuda' else 'gloo', init_method='env://', rank=0, world_size=1)
            from utils.dist import fsdp_wrap, fsdp_state_dict
            model = fsdp_wrap(model)
        except Exception as e:
            print(f'[WARN] FSDP init failed ({e}); running single process.')
            args.mode = 'single'

    metrics = {}
    for epoch in range(1, args.epochs+1):
        loss, acc, ips = train_one_epoch(model, loader, criterion, optimizer, device, use_amp=args.amp)
        print(f'Epoch {epoch}: loss={loss:.4f} acc={acc:.3f} throughput={ips:.1f} samples/s')
        metrics[f'epoch_{epoch}'] = {'loss': loss, 'acc': acc, 'throughput_sps': ips}

    # Save checkpoint
    ckpt_path = 'artifacts/model.pt'
    if args.mode == 'fsdp' and hasattr(model, 'state_dict'):
        try:
            from utils.dist import fsdp_state_dict
            sd = fsdp_state_dict(model)
            torch.save(sd, ckpt_path)
        except Exception as e:
            print(f'[WARN] FSDP state_dict failed ({e}); saving local model state.')
            torch.save(model.state_dict(), ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)

    save_metrics('artifacts/train_report.json', {'mode': args.mode, 'amp': bool(args.amp), **metrics})
    print('Saved:', ckpt_path, 'and artifacts/train_report.json')

if __name__ == '__main__':
    main()
