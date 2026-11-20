import torch

# Feature group definitions for ablation study
FEATURE_GROUPS = {
    "density":          [0],
    "radial":           [1, 2, 3],
    "bbox":             [4, 5, 6, 7],
    "eigenvalues":      [8, 9, 10],
    "shape":            [11, 12, 13, 14, 15, 16, 17, 18],
    "normal":           [19, 20, 21],
}


def compute_geom_features(points, eps=1e-6, ablation_mask=None):
    """Compute fixed-length geometry descriptors for point cloud patches.

    Args:
        points: Tensor with shape ``[N, 3]`` or ``[B, N, 3]`` containing xyz
            coordinates. NaN/Inf values are replaced with zeros.
        eps: Small constant used for divisions and logarithms.
        ablation_mask: Optional list of feature group names to zero out.
            Valid group names: 'density', 'radial', 'bbox', 'eigenvalues',
            'shape', 'normal'. Example: ['density', 'normal'] zeros out
            features 0 and 19-21. None means use all features (default).

    Returns:
        Tensor with shape ``[F]`` for unbatched input or ``[B, F]`` for batched
        input. The 22 descriptors are density, radial spread, bounding-box
        size, covariance eigenvalues, and eigenvalue-derived shape measures
        such as linearity, planarity, scattering, curvature, anisotropy,
        omnivariance, eigenentropy, surface variation, and dominant normal.
    """
    unbatched = points.dim() == 2
    if unbatched:
        points = points.unsqueeze(0)
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError("points must have shape [N, 3] or [B, N, 3]")

    points = torch.nan_to_num(points.float(), nan=0.0, posinf=0.0, neginf=0.0)
    batch_size, point_count, _ = points.shape

    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid
    radius = torch.linalg.norm(centered, dim=-1)
    mean_radius = radius.mean(dim=1, keepdim=True)
    std_radius = radius.std(dim=1, keepdim=True, unbiased=False)
    max_radius = radius.max(dim=1, keepdim=True).values

    bbox_min = points.min(dim=1).values
    bbox_max = points.max(dim=1).values
    bbox_size = bbox_max - bbox_min
    bbox_diag = torch.linalg.norm(bbox_size, dim=1, keepdim=True).clamp_min(eps)
    density = torch.full(
        (batch_size, 1),
        float(point_count),
        device=points.device,
        dtype=points.dtype,
    ) / (bbox_size.prod(dim=1, keepdim=True).abs() + eps)

    cov = centered.transpose(1, 2).matmul(centered) / max(point_count - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eps)
    eigvals = torch.flip(eigvals, dims=[1])
    eigvecs = torch.flip(eigvecs, dims=[2])
    eig_sum = eigvals.sum(dim=1, keepdim=True).clamp_min(eps)
    eig_norm = eigvals / eig_sum

    l1, l2, l3 = eigvals[:, 0:1], eigvals[:, 1:2], eigvals[:, 2:3]
    linearity = (l1 - l2) / l1.clamp_min(eps)
    planarity = (l2 - l3) / l1.clamp_min(eps)
    scattering = l3 / l1.clamp_min(eps)
    curvature = l3 / (l1 + l2 + l3).clamp_min(eps)
    anisotropy = (l1 - l3) / l1.clamp_min(eps)
    omnivariance = (l1 * l2 * l3).clamp_min(eps).pow(1.0 / 3.0)
    eigenentropy = -(eig_norm * torch.log(eig_norm.clamp_min(eps))).sum(dim=1, keepdim=True)
    surface_variation = curvature
    normal = eigvecs[:, :, 2].abs()

    features = torch.cat(
        [
            density,           # 0
            mean_radius,       # 1
            std_radius,        # 2
            max_radius,        # 3
            bbox_size / bbox_diag,  # 4-6
            bbox_diag,         # 7
            eig_norm,          # 8-10
            linearity,         # 11
            planarity,         # 12
            scattering,        # 13
            curvature,         # 14
            anisotropy,        # 15
            omnivariance,      # 16
            eigenentropy,      # 17
            surface_variation, # 18
            normal,            # 19-21
        ],
        dim=1,
    )

    # Ablation: zero out specified feature groups
    if ablation_mask is not None:
        for group_name in ablation_mask:
            if group_name not in FEATURE_GROUPS:
                raise ValueError(
                    f"Unknown feature group '{group_name}'. "
                    f"Valid groups: {list(FEATURE_GROUPS.keys())}"
                )
            for idx in FEATURE_GROUPS[group_name]:
                features[:, idx] = 0.0

    return features.squeeze(0) if unbatched else features
