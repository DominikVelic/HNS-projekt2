from pathlib import Path
import time
from sklearn.metrics import classification_report, confusion_matrix
import torch
from hns_projekt2.train.config import choose_device, seed_everything
import pandas as pd

from hns_projekt2.train.loaders import make_loaders
from hns_projekt2.train.models import MODELS_KEYS, build_model


def run_epoch(model: torch.nn.Module, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"

    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())

    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return total_loss / len(labels), (preds == labels).mean(), preds, labels

def train_model(model_key: str, data_root: Path, out_dir: Path, epochs: int, batch_size: int,
                lr: float, num_workers: int, seed: int):
    seed_everything(seed)
    device = choose_device()
    name = MODELS_KEYS[model_key]
    print(f"\n===Model: {name} on device: {device} ===")
    model, data_cfg = build_model(model_key, pretrained=True, num_classes=2)
    model = model.to(device)
    loaders, classes = make_loaders(data_root, data_cfg, batch_size, num_workers)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    out = out_dir
    out.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer, scaler)
        va_loss, va_acc, _, _ = run_epoch(model, loaders["val"], device, criterion)
        scheduler.step()

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=time.time() - t0))
        print(f"epoch {epoch}/{epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}  ({time.time() - t0:.0f}s)")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), out / "best.pt")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    if "test" in loaders:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        _, te_acc, preds, labels = run_epoch(model, loaders["test"], device, criterion)
        report = classification_report(labels, preds, target_names=classes, digits=4)
        cm = confusion_matrix(labels, preds)
        print(f"test acc {te_acc:.4f}\n{report}\n{cm}")
        (out / "test_report.txt").write_text(f"{report}\n{cm}\n")


