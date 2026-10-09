"""Clustering primitives shared by training and analysis stages."""

from __future__ import annotations

from typing import Tuple

import torch


def run_kmeans(
    x: torch.Tensor, k: int, iters: int, restarts: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if k < 2:
        raise ValueError("k must be at least 2.")
    if x.shape[0] < k:
        raise ValueError(f"Cannot cluster {x.shape[0]} samples into {k} clusters.")
    generator = torch.Generator().manual_seed(int(seed))
    best_labels = None
    best_centroids = None
    best_inertia = float("inf")
    x = x.float()
    for _ in range(max(1, restarts)):
        init_idx = torch.randperm(x.shape[0], generator=generator)[:k]
        centroids = x[init_idx].clone()
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        for _ in range(max(1, iters)):
            distances = torch.cdist(x, centroids)
            new_labels = distances.argmin(dim=1)
            new_centroids = []
            for cluster_id in range(k):
                mask = new_labels == cluster_id
                if mask.any():
                    new_centroids.append(x[mask].mean(dim=0))
                else:
                    replacement = torch.randint(
                        0, x.shape[0], (1,), generator=generator
                    ).item()
                    new_centroids.append(x[replacement])
            next_centroids = torch.stack(new_centroids, dim=0)
            if torch.equal(labels, new_labels):
                centroids = next_centroids
                labels = new_labels
                break
            labels = new_labels
            centroids = next_centroids
        inertia = (x - centroids[labels]).pow(2).sum(dim=1).mean().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centroids = centroids.clone()
    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, float(best_inertia)


def centroid_silhouette_score(
    x: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor
) -> float:
    if centroids.shape[0] < 2 or x.shape[0] == 0:
        return 0.0
    distances = torch.cdist(x.float(), centroids.float())
    row_indices = torch.arange(x.shape[0])
    assigned = distances[row_indices, labels.long()]
    masked = distances.clone()
    masked[row_indices, labels.long()] = float("inf")
    nearest_other = masked.min(dim=1).values
    denominator = torch.maximum(assigned, nearest_other).clamp_min(1e-6)
    return float(((nearest_other - assigned) / denominator).mean().item())
