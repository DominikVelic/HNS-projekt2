from pathlib import Path
import warnings

from PIL import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from hns_projekt2.train.transform import make_transforms

def try_gradcam(model, key, paths, labels, preds, class_names, device, outdir, count=8):
    """Basic Grad-CAM for architectures with a convolutional layer."""
    target = None
    if key == "resnet18_cbam":
        # The last Conv2d here is CBAM's spatial-attention conv; use the backbone's final stage instead.
        target = model.backbone.layer4
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target = module
                break
    if target is None:
        print(f"[Grad-CAM] {key}: nenašla sa konvolučná vrstva; preskakujem.")
        return
    _, eval_tf = make_transforms(False)
    os.makedirs(outdir, exist_ok=True)
    activations, gradients = {}, {}

    def forward_hook(_m, _i, output):
        activations["value"] = output

    def backward_hook(_m, _gi, go):
        gradients["value"] = go[0]

    h1 = target.register_forward_hook(forward_hook)
    h2 = target.register_full_backward_hook(backward_hook)
    try:
        # Half correctly and half incorrectly classified test samples; if there are too few
        # misclassifications, fill the rest with correct ones.
        correct = np.flatnonzero(preds == labels)
        wrong = np.flatnonzero(preds != labels)
        n_wrong = min(len(wrong), count // 2)
        chosen = list(wrong[:n_wrong]) + list(correct[:count - n_wrong])
        for n, idx in enumerate(chosen):
            status = "ok" if preds[idx] == labels[idx] else "wrong"
            with Image.open(paths[idx]) as im:
                pil = im.convert("RGB")
            x = eval_tf(pil).unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            pred = int(logits.argmax(1).item())
            logits[0, pred].backward()
            if "value" not in activations or "value" not in gradients:
                continue
            acts, grads = activations["value"], gradients["value"]
            if acts.ndim != 4:
                continue
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(224, 224),
                                                  mode="bilinear", align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pil.resize((224, 224)))
            ax.imshow(cam, cmap="jet", alpha=0.4)
            ax.set_title(f"Skutočná: {class_names[int(labels[idx])]}; predikcia: {
                         class_names[pred]}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(Path(outdir) /
                        f"gradcam_{n:02d}_{status}.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Grad-CAM sa nepodaril pre {key}: {exc}")
    finally:
        h1.remove()
        h2.remove()
