# ternary-tensor: Tensor operations for {-1, 0, +1} multi-dimensional arrays

An N-dimensional tensor library for ternary values, with dense and sparse storage, matrix operations, broadcasting, contraction, and CP-style decomposition.

## Why This Exists

Ternary neural networks (like XNOR-Net and its successors) represent weights and activations as {-1, 0, +1} to replace multiplications with simple sign operations. Standard tensor libraries waste memory and compute treating ternary as a special case of float. This library provides native ternary tensor operations — no floats, no wasted bits.

## Core Concepts

**Trit** — A ternary digit, analogous to a bit. One of three values: Neg (-1), Zero (0), or Pos (+1).

**TernaryTensor** — An N-dimensional dense array of trits, stored in row-major order. Supports indexing, element-wise operations, broadcasting, and contraction.

**SparseTernaryTensor** — A ternary tensor that stores only non-zero entries in a HashMap. Efficient when most values are Zero.

**Contraction** — Summing a tensor along specified axes, reducing dimensionality. The sum is clamped back to ternary range.

**Broadcasting** — Stretching a tensor with dimensions of size 1 to match a larger shape, following NumPy conventions (right-aligned).

**CP decomposition** — CANDECOMP/PARAFAC: expressing a tensor as a sum of rank-1 outer products. Each component has a weight and factor vectors for each mode.

**Matrix chain multiplication** — Multiplying a sequence of 2D tensors left-to-right, useful for neural network layer composition.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-tensor = "0.1"
```

```rust
use ternary_tensor::*;

// Create a 2x3 tensor
let mut t = TernaryTensor::zeros(vec![2, 3]);
t.set(&[0, 0], Trit::Pos);
t.set(&[1, 2], Trit::Neg);

// Matrix multiplication
let a = TernaryTensor::matrix(2, 2, vec![
    Trit::Pos, Trit::Neg,
    Trit::Zero, Trit::Pos,
]);
let b = TernaryTensor::matrix(2, 2, vec![
    Trit::Pos, Trit::Zero,
    Trit::Neg, Trit::Pos,
]);
let c = matmul(&a, &b);
assert_eq!(c.get(&[0, 0]), Trit::Pos); // 1*1 + (-1)*(-1) = 2, clamped to 1
```

## API Overview

| Type / Function | Description |
|----------------|-------------|
| `Trit` | Ternary value: `Neg`, `Zero`, `Pos` |
| `TernaryTensor` | Dense N-dimensional tensor of trits |
| `SparseTernaryTensor` | HashMap-backed sparse ternary tensor |
| `matmul(a, b)` | Matrix multiplication for 2D tensors |
| `chain_multiply(mats)` | Left-to-right matrix chain multiplication |
| `contract(tensor, axes)` | Sum along specified axes (dimensionality reduction) |
| `cp_decompose(tensor, rank)` | CP-style decomposition into rank-1 components |
| `CPDecomposition` | Result of CP decomposition: weights + factor vectors |

## How It Works

Dense tensors store data in a flat `Vec<Trit>` in row-major (C-style) order. Indexing converts multi-dimensional coordinates to a flat offset using stride computation. Element-wise operations iterate the flat array directly.

Matrix multiplication follows the standard triple-loop algorithm. Products are summed as i32 and clamped to {-1, 0, +1} before storing. This means large-magnitude intermediate results are lost — by design, since ternary networks only care about sign.

Broadcasting pads the source shape with leading 1s, then for each output position, maps to the source position by replacing broadcast dimensions' indices with 0.

Sparse tensors use a `HashMap<Vec<usize>, Trit>`, which is not the most cache-friendly but is simple and correct for moderate sizes. Zero-valued entries are never stored.

## Known Limitations

- **Clamping loss**: Matrix multiply and contraction clamp intermediate sums to {-1, 0, +1}. Information about magnitude (e.g., "3 Pos values summed") is discarded. This is correct for sign-only ternary networks but wrong for applications needing full integer sums.
- **Sparse performance**: The HashMap-based sparse storage has O(log n) access per element. For very sparse very large tensors, a CSR/CSC format would be faster.
- **No GPU support**: All operations are single-threaded CPU. For production ternary neural networks, you'd want GPU kernels.
- **CP decomposition is simplified**: The `cp_decompose` function uses a sign-based heuristic, not an alternating least squares (ALS) optimizer. It gives a structural decomposition, not an optimized low-rank approximation.

## Use Cases

- **Ternary neural network inference** — Forward pass through layers with {-1, 0, +1} weights and activations, replacing float matmul with ternary matmul.
- **Compressed model storage** — Sparse ternary tensors for storing pruned binary/ternary network weights.
- **Multi-dimensional ternary signal processing** — Contraction and broadcasting for filtering and transforming ternary sensor data.
- **Game theory payoff tensors** — N-player games with ternary outcomes (win/lose/draw) stored as tensors.

## Ecosystem Context

Part of the SuperInstance ternary computing ecosystem. Related crates:

- `ternary-matrix` — 2D-only ternary matrix operations (simpler API for matrix-focused work)
- `ternary-transform` — Higher-level transforms (FFT, etc.) built on this tensor type
- `ternary-compression` — Compression codecs that use sparse ternary tensors

This crate is the foundational data structure for multi-dimensional ternary computation.

## License

MIT

## See Also
- **ternary-matrix** — related
- **ternary-ring** — related
- **ternary-network** — related
- **ternary-graph** — related
- **ternary-database** — related
- **ternary-projection** — related

