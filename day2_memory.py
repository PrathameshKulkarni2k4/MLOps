import numpy as np
import time

# ---------- Part 1: Inspect Strides & Contiguity ----------
arr_c = np.ones((1000, 500), order="C", dtype=np.float64)
print(f"C-order strides: {arr_c.strides}")
print(f"C_CONTIGUOUS: {arr_c.flags['C_CONTIGUOUS']}")

arr_t = arr_c.T  # Transpose returns a view, swapping strides
print(f"Transposed strides: {arr_t.strides}")
print(f"Transposed C_CONTIGUOUS: {arr_t.flags['C_CONTIGUOUS']}")
print(f"Transposed F_CONTIGUOUS: {arr_t.flags['F_CONTIGUOUS']}")
print(f"Shares memory: {np.shares_memory(arr_c, arr_t)}")

# ---------- Part 2: Benchmark Cache Locality ----------
def benchmark_layouts():
    size = 8000   # 8000 x 8000 float64 approx 512 MB; reduce if memory error
    matrix_c = np.random.randn(size, size).astype(np.float64)  # C-order
    matrix_f = np.asfortranarray(matrix_c)                     # F-order copy

    # C-order row-wise sum (contiguous, cache-friendly)
    start = time.perf_counter()
    _ = np.sum(matrix_c, axis=1)
    time_c_row = time.perf_counter() - start

    # C-order column-wise sum (non-contiguous, strided)
    start = time.perf_counter()
    _ = np.sum(matrix_c, axis=0)
    time_c_col = time.perf_counter() - start

    # F-order row-wise sum (non-contiguous)
    start = time.perf_counter()
    _ = np.sum(matrix_f, axis=1)
    time_f_row = time.perf_counter() - start

    # F-order column-wise sum (contiguous, cache-friendly)
    start = time.perf_counter()
    _ = np.sum(matrix_f, axis=0)
    time_f_col = time.perf_counter() - start

    print(f"[C-Array] Row sum (contiguous):    {time_c_row:.4f}s")
    print(f"[C-Array] Column sum (non-contig): {time_c_col:.4f}s")
    print(f"[F-Array] Row sum (non-contig):    {time_f_row:.4f}s")
    print(f"[F-Array] Column sum (contiguous): {time_f_col:.4f}s")

benchmark_layouts()

# ---------- Part 3: Vectorized Standardization ----------
def standardize_features(X: np.ndarray) -> np.ndarray:
    """Vectorized Z-score standardization across columns (axis 0)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    return (X - mean) / std

# Test the function
X = np.random.randn(100, 5) * 10 + 5   # mean=5, std=10
X_std = standardize_features(X)
print(f"Column means after standardization: {X_std.mean(axis=0)}")
print(f"Column stds after standardization:  {X_std.std(axis=0)}")
