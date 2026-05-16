# RV-Sparse: Coding Challenge

Implementation of `sparse_multiply` — a zero-allocation sparse matrix-vector product using **Compressed Sparse Row (CSR)** format.

## Build & Run

```bash
gcc -o run challenge.c -lm
./run
```

Expected output:

```
Iter  0 [ ...x..., density=..., nnz=...]: PASS (Max error: 0.00e+00)
...
All tests passed! (100/100 iterations passed)
```

## Function Signature

```c
void sparse_multiply(
    int rows,
    int cols,
    const double* A,        // row-major input matrix (rows x cols)
    const double* x,        // input vector (length cols)
    int*    out_nnz,        // output: number of non-zero elements found
    double* values,         // CSR buffer: non-zero values
    int*    col_indices,    // CSR buffer: column indices of non-zeros
    int*    row_ptrs,       // CSR buffer: row pointer array (length rows+1)
    double* y               // output vector (length rows)
);
```

All buffers are **caller-allocated**. The function performs **zero dynamic memory allocation**.

## Algorithm

**Phase 1 — CSR Extraction** `O(rows × cols)`:

Scan `A` row by row. For each non-zero element, record its value and column index into `values[]` and `col_indices[]`. Track the start of each row in `row_ptrs[i]`, with `row_ptrs[rows]` as a sentinel equal to `nnz`.

```
nnz = 0
for i in [0, rows):
    row_ptrs[i] = nnz
    for j in [0, cols):
        if A[i][j] != 0:
            values[nnz]      = A[i][j]
            col_indices[nnz] = j
            nnz++
row_ptrs[rows] = nnz
*out_nnz = nnz
```

**Phase 2 — SpMV** `O(nnz)`:

For each row, accumulate the dot product over only the stored non-zeros.

```
for i in [0, rows):
    y[i] = sum of values[k] * x[col_indices[k]]
            for k in [row_ptrs[i], row_ptrs[i+1])
```

## Complexity

| Phase | Time | Extra Space |
|---|---|---|
| CSR extraction | O(rows × cols) | O(1) — buffers are caller's |
| SpMV | O(nnz) | O(1) |

## Test Harness

The provided harness runs **100 randomised iterations**, each with:
- Matrix dimensions: rows and cols independently in `[5, 45]`
- Density: uniformly sampled in `[0.05, 0.40]`
- Values: uniformly sampled in `[-10, 10]`
- Tolerance: mixed absolute/relative — `1e-7 + 1e-7 * |y_ref[i]|`

Result: **100/100 PASS, max error 0.00e+00** across all iterations.
