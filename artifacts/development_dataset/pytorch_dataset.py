"""Minimal PyTorch reader for the Idea-1 hybrid development dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "validation", "all"]
FeatureTensor = Literal["x", "x_normalized", "x_design"]


class Idea1HybridDataset(Dataset):
    """Return masked multi-task targets without converting censoring to zero."""

    def __init__(
        self,
        directory: str | Path,
        *,
        split: Split = "train",
        feature_tensor: FeatureTensor = "x_normalized",
    ) -> None:
        archive = np.load(Path(directory) / "tensors.npz", allow_pickle=False)
        split_code = archive["split_code"]
        if split == "train":
            indices = np.flatnonzero(split_code == 0)
        elif split == "validation":
            indices = np.flatnonzero(split_code == 1)
        elif split == "all":
            indices = np.arange(split_code.size)
        else:
            raise ValueError(f"unsupported split: {split}")
        if feature_tensor not in {"x", "x_normalized", "x_design"}:
            raise ValueError(f"unsupported feature tensor: {feature_tensor}")

        self.indices = indices
        self.x = archive[feature_tensor]
        self.domain_complete = archive["domain_complete"]
        self.response_mask = archive["response_mask"]
        self.component_support = archive["component_support"]
        self.component_hits = archive["component_hits"]
        self.component_penetrations = archive["component_penetrations"]
        self.component_energy_j = archive["component_energy_j"]
        self.component_severity = archive["component_severity"]
        self.tree_outputs = archive["tree_outputs"]
        self.summary_targets = archive["summary_targets"]
        self.candidate_group_id = archive["candidate_group_id"]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        return {
            "x": torch.from_numpy(self.x[index]),
            "domain_complete": torch.tensor(
                float(self.domain_complete[index]), dtype=torch.float32
            ),
            "response_mask": torch.tensor(
                bool(self.response_mask[index]), dtype=torch.bool
            ),
            "component_support": torch.from_numpy(self.component_support[index]),
            "component_hits": torch.from_numpy(self.component_hits[index]),
            "component_penetrations": torch.from_numpy(
                self.component_penetrations[index]
            ),
            "component_energy_j": torch.from_numpy(self.component_energy_j[index]),
            "component_severity": torch.from_numpy(self.component_severity[index]),
            "tree_outputs": torch.from_numpy(self.tree_outputs[index]),
            "summary_targets": torch.from_numpy(self.summary_targets[index]),
            "candidate_group_id": str(self.candidate_group_id[index]),
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train = Idea1HybridDataset(root, split="train")
    validation = Idea1HybridDataset(root, split="validation")
    first = train[0]
    print(
        {
            "train": len(train),
            "validation": len(validation),
            "x_shape": tuple(first["x"].shape),
            "component_shape": tuple(first["component_support"].shape),
            "tree_shape": tuple(first["tree_outputs"].shape),
        }
    )
