from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(paths, labels, class_names, outdir, seed=42):
    indices = np.arange(len(paths))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
    )

    manifest = []

    for split, idxs in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx)
    ]:
        manifest += [
            {
                "path": paths[i],
                "label_id": int(labels[i]),
                "class_name": class_names[labels[i]],
                "split": split
            }
            for i in idxs
        ]

    pd.DataFrame(manifest).to_csv(
        Path(outdir) / "split_manifest.csv",
        index=False
    )
    return train_idx, val_idx, test_idx
