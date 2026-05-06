import numpy as np
import torch as th


def to_torch_gpu(X, device="cuda", dtype=th.float32):
    if isinstance(X, np.ndarray):
        X = th.from_numpy(X)
    elif not th.is_tensor(X):
        raise TypeError(f"Unsupported type: {type(X)}")

    return X.to(device=device, dtype=dtype, non_blocking=True)


def center_features(X):
    return X - X.mean(dim=0, keepdim=True)


def _precompute_cka_terms(reps, ordered_keys, device="cuda"):
    """
    Per ogni matrice X precompute:
    - Xc = X centrata
    - self_norm = ||Xc^T Xc||_F
    """
    cached = {}

    for key in ordered_keys:
        X = to_torch_gpu(reps[key], device=device, dtype=th.float32)
        Xc = center_features(X)

        xTx = Xc.T @ Xc
        self_norm = th.sqrt((xTx * xTx).sum())

        cached[key] = {
            "Xc": Xc,
            "self_norm": self_norm,
        }

    return cached


def linear_cka_from_centered(Xc, Yc, norm_x, norm_y, eps=1e-12):
    xTy = Xc.T @ Yc
    hsic = (xTy * xTy).sum()
    return hsic / (norm_x * norm_y + eps)


def compute_cka_matrix(reps, ordered_keys, device="cuda"):
    n = len(ordered_keys)
    C = th.zeros((n, n), dtype=th.float32, device=device)

    cached = _precompute_cka_terms(
        reps=reps,
        ordered_keys=ordered_keys,
        device=device,
    )

    for i, key_i in enumerate(ordered_keys):
        Xi_c = cached[key_i]["Xc"]
        norm_i = cached[key_i]["self_norm"]

        for j in range(i, n):
            key_j = ordered_keys[j]
            Xj_c = cached[key_j]["Xc"]
            norm_j = cached[key_j]["self_norm"]

            if Xi_c.shape[0] != Xj_c.shape[0]:
                raise ValueError(
                    f"CKA requires same number of rows: "
                    f"{key_i} -> {tuple(Xi_c.shape)}, "
                    f"{key_j} -> {tuple(Xj_c.shape)}"
                )

            score = linear_cka_from_centered(
                Xc=Xi_c,
                Yc=Xj_c,
                norm_x=norm_i,
                norm_y=norm_j,
            )

            C[i, j] = score
            C[j, i] = score

    return C.detach().cpu().numpy()

import time
import numpy as np
import torch as th


def compute_cka_matrix_profiled(reps, ordered_keys, device="cuda"):
    n = len(ordered_keys)
    C = th.zeros((n, n), dtype=th.float32, device=device)

    if device.startswith("cuda"):
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)

    t0 = time.perf_counter()
    cached = _precompute_cka_terms(
        reps=reps,
        ordered_keys=ordered_keys,
        device=device,
    )
    if device.startswith("cuda"):
        th.cuda.synchronize(device)
    t1 = time.perf_counter()

    print(f"[CKA] precompute_time = {t1 - t0:.2f}s")
    if device.startswith("cuda"):
        peak_mb = th.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[CKA] peak_mem_after_precompute = {peak_mb:.2f} MB")

    for i, key_i in enumerate(ordered_keys):
        row_t0 = time.perf_counter()

        Xi_c = cached[key_i]["Xc"]
        norm_i = cached[key_i]["self_norm"]

        for j in range(i, n):
            key_j = ordered_keys[j]
            Xj_c = cached[key_j]["Xc"]
            norm_j = cached[key_j]["self_norm"]

            score = linear_cka_from_centered(
                Xc=Xi_c,
                Yc=Xj_c,
                norm_x=norm_i,
                norm_y=norm_j,
            )

            C[i, j] = score
            C[j, i] = score

        if device.startswith("cuda"):
            th.cuda.synchronize(device)

        row_t1 = time.perf_counter()
        print(f"[CKA] row {i+1}/{n} done in {row_t1 - row_t0:.2f}s | key={key_i}")

    if device.startswith("cuda"):
        total_peak_mb = th.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[CKA] final_peak_mem = {total_peak_mb:.2f} MB")

    return C.detach().cpu().numpy()