#![forbid(unsafe_code)]

//! Tensor operations for ternary multi-dimensional arrays.
//!
//! Provides N-dimensional tensors over {-1, 0, +1}, with contraction,
//! broadcasting, element-wise operations, matrix chain multiplication,
//! CP-style decomposition, and sparse storage.

use std::collections::HashMap;

/// A ternary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Trit {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn multiply(self, other: Trit) -> Trit {
        match (self, other) {
            (Trit::Zero, _) | (_, Trit::Zero) => Trit::Zero,
            (Trit::Pos, o) | (o, Trit::Pos) => o,
            (Trit::Neg, Trit::Neg) => Trit::Pos,
        }
    }

    pub fn add(self, other: Trit) -> Trit {
        let sum = self.to_i8() + other.to_i8();
        Trit::from_i8(sum.clamp(-1, 1)).unwrap_or(Trit::Zero)
    }
}

/// Index into a tensor: a list of axis positions.
pub type TensorIndex = Vec<usize>;

/// An N-dimensional tensor of ternary values stored densely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TernaryTensor {
    /// Shape of each dimension.
    pub shape: Vec<usize>,
    /// Data stored in row-major order.
    pub data: Vec<Trit>,
}

impl TernaryTensor {
    /// Create a new tensor with the given shape, filled with Trit::Zero.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            shape,
            data: vec![Trit::Zero; total],
        }
    }

    /// Create a tensor filled with a single value.
    pub fn filled(shape: Vec<usize>, value: Trit) -> Self {
        let total: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; total],
        }
    }

    /// Create a tensor from raw data. Panics if data length doesn't match shape.
    pub fn from_vec(shape: Vec<usize>, data: Vec<Trit>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Data length doesn't match shape");
        Self { shape, data }
    }

    /// Create a 2D tensor (matrix) from a flat slice in row-major order.
    pub fn matrix(rows: usize, cols: usize, data: Vec<Trit>) -> Self {
        Self::from_vec(vec![rows, cols], data)
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn flat_index(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.shape.len(), "Index rank mismatch");
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..self.shape.len()).rev() {
            assert!(index[i] < self.shape[i], "Index out of bounds on axis {}", i);
            flat += index[i] * stride;
            stride *= self.shape[i];
        }
        flat
    }

    pub fn get(&self, index: &[usize]) -> Trit {
        self.data[self.flat_index(index)]
    }

    pub fn set(&mut self, index: &[usize], value: Trit) {
        let fi = self.flat_index(index);
        self.data[fi] = value;
    }

    /// Element-wise unary operation.
    pub fn map<F: Fn(Trit) -> Trit>(&self, f: F) -> TernaryTensor {
        TernaryTensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&t| f(t)).collect(),
        }
    }

    /// Element-wise binary operation on two tensors of the same shape.
    pub fn elementwise<F: Fn(Trit, Trit) -> Trit>(&self, other: &TernaryTensor, f: F) -> TernaryTensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in elementwise op");
        TernaryTensor {
            shape: self.shape.clone(),
            data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect(),
        }
    }

    /// Element-wise addition (clamped to ternary range).
    pub fn add(&self, other: &TernaryTensor) -> TernaryTensor {
        self.elementwise(other, Trit::add)
    }

    /// Element-wise multiplication.
    pub fn multiply(&self, other: &TernaryTensor) -> TernaryTensor {
        self.elementwise(other, Trit::multiply)
    }

    /// Negate all elements: Pos ↔ Neg, Zero stays Zero.
    pub fn negate(&self) -> TernaryTensor {
        self.map(|t| match t {
            Trit::Pos => Trit::Neg,
            Trit::Neg => Trit::Pos,
            Trit::Zero => Trit::Zero,
        })
    }

    /// Broadcast the tensor to a target shape following numpy-like rules.
    /// Dimensions of size 1 are stretched; ranks are aligned from the right.
    pub fn broadcast(&self, target_shape: &[usize]) -> TernaryTensor {
        let self_rank = self.shape.len();
        let target_rank = target_shape.len();

        // Pad self shape with leading 1s
        let mut padded = vec![1usize; target_rank - self_rank];
        padded.extend_from_slice(&self.shape);

        // Validate: each dim must be 1 or match target
        for i in 0..target_rank {
            assert!(
                padded[i] == 1 || padded[i] == target_shape[i],
                "Cannot broadcast dimension {} from {} to {}",
                i, padded[i], target_shape[i]
            );
        }

        let total: usize = target_shape.iter().product();
        let mut result = vec![Trit::Zero; total];

        // For each output index, compute the source index
        let mut out_index = vec![0usize; target_rank];
        for flat in 0..total {
            // Compute multi-index
            let mut remaining = flat;
            for d in (0..target_rank).rev() {
                out_index[d] = remaining % target_shape[d];
                remaining /= target_shape[d];
            }
            // Map to source index
            let mut src_flat = 0usize;
            let mut stride = 1usize;
            for d in (0..self_rank).rev() {
                let src_d = out_index[d + (target_rank - self_rank)];
                let src_idx = if padded[d + (target_rank - self_rank)] == 1 {
                    0
                } else {
                    src_d
                };
                src_flat += src_idx * stride;
                stride *= self.shape[d];
            }
            result[flat] = self.data[src_flat];
        }

        TernaryTensor {
            shape: target_shape.to_vec(),
            data: result,
        }
    }

    /// Sum all elements, returning an i32 (not clamped to ternary).
    pub fn sum(&self) -> i32 {
        self.data.iter().map(|t| t.to_i8() as i32).sum()
    }

    /// Count non-zero elements.
    pub fn count_nonzero(&self) -> usize {
        self.data.iter().filter(|&&t| t != Trit::Zero).count()
    }
}

/// Matrix multiplication for 2D ternary tensors.
/// Product values are clamped to {-1, 0, +1}.
pub fn matmul(a: &TernaryTensor, b: &TernaryTensor) -> TernaryTensor {
    assert_eq!(a.rank(), 2, "matmul requires 2D tensors");
    assert_eq!(b.rank(), 2, "matmul requires 2D tensors");
    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k1, k2, "Inner dimensions must match");

    let mut result = vec![Trit::Zero; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum: i32 = 0;
            for k in 0..k1 {
                sum += a.data[i * k1 + k].to_i8() as i32 * b.data[k * n + j].to_i8() as i32;
            }
            result[i * n + j] = Trit::from_i8(sum.clamp(-1, 1) as i8).unwrap_or(Trit::Zero);
        }
    }
    TernaryTensor {
        shape: vec![m, n],
        data: result,
    }
}

/// Chain multiply a sequence of 2D matrices.
/// Multiplies left to right: ((A * B) * C) * ...
pub fn chain_multiply(matrices: &[TernaryTensor]) -> TernaryTensor {
    assert!(!matrices.is_empty(), "Need at least one matrix");
    let mut result = matrices[0].clone();
    for mat in &matrices[1..] {
        result = matmul(&result, mat);
    }
    result
}

/// Contract a tensor along specified axes by summing (ternary-clamped).
///
/// `axes` specifies which dimensions to reduce. Each reduced dimension
/// is summed and the dimension is removed from the output.
pub fn contract(tensor: &TernaryTensor, axes: &[usize]) -> TernaryTensor {
    if axes.is_empty() {
        return tensor.clone();
    }

    let mut new_shape: Vec<usize> = Vec::new();
    let axis_set: std::collections::HashSet<usize> = axes.iter().copied().collect();
    for (i, &dim) in tensor.shape.iter().enumerate() {
        if !axis_set.contains(&i) {
            new_shape.push(dim);
        }
    }
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    let total: usize = new_shape.iter().product();
    let mut result = vec![0i32; total];

    // Iterate over all elements
    let mut index = vec![0usize; tensor.shape.len()];
    for flat in 0..tensor.data.len() {
        // Compute multi-index
        let mut remaining = flat;
        for d in (0..tensor.shape.len()).rev() {
            index[d] = remaining % tensor.shape[d];
            remaining /= tensor.shape[d];
        }
        // Map to output index
        let mut out_flat = 0usize;
        let mut stride = 1usize;
        for d in (0..tensor.shape.len()).rev() {
            if !axis_set.contains(&d) {
                out_flat += index[d] * stride;
                stride *= tensor.shape[d];
            }
        }
        result[out_flat] += tensor.data[flat].to_i8() as i32;
    }

    TernaryTensor {
        shape: new_shape,
        data: result.iter().map(|&v| Trit::from_i8(v.clamp(-1, 1) as i8).unwrap_or(Trit::Zero)).collect(),
    }
}

/// A sparse ternary tensor storing only non-zero values in a HashMap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTernaryTensor {
    pub shape: Vec<usize>,
    pub entries: HashMap<TensorIndex, Trit>,
}

impl SparseTernaryTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            entries: HashMap::new(),
        }
    }

    pub fn from_dense(tensor: &TernaryTensor) -> Self {
        let mut sparse = Self::new(tensor.shape.clone());
        let mut index = vec![0usize; tensor.shape.len()];
        for (flat, &val) in tensor.data.iter().enumerate() {
            if val != Trit::Zero {
                let mut remaining = flat;
                for d in (0..tensor.shape.len()).rev() {
                    index[d] = remaining % tensor.shape[d];
                    remaining /= tensor.shape[d];
                }
                sparse.entries.insert(index.clone(), val);
            }
        }
        sparse
    }

    pub fn to_dense(&self) -> TernaryTensor {
        let total: usize = self.shape.iter().product();
        let mut tensor = TernaryTensor::zeros(self.shape.clone());
        for (index, &val) in &self.entries {
            tensor.set(index, val);
        }
        tensor
    }

    pub fn get(&self, index: &[usize]) -> Trit {
        self.entries.get(index).copied().unwrap_or(Trit::Zero)
    }

    pub fn set(&mut self, index: TensorIndex, value: Trit) {
        if value == Trit::Zero {
            self.entries.remove(&index);
        } else {
            self.entries.insert(index, value);
        }
    }

    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn density(&self) -> f64 {
        let total: usize = self.shape.iter().product();
        if total == 0 { return 0.0; }
        self.entries.len() as f64 / total as f64
    }

    /// Element-wise add two sparse tensors.
    pub fn add(&self, other: &SparseTernaryTensor) -> SparseTernaryTensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let mut result = self.clone();
        for (index, &val) in &other.entries {
            let current = result.get(index);
            let new_val = current.add(val);
            if new_val == Trit::Zero {
                result.entries.remove(index);
            } else {
                result.entries.insert(index.clone(), new_val);
            }
        }
        result
    }
}

/// CP-style tensor decomposition result.
///
/// Decomposes a tensor into a sum of rank-1 outer products.
/// Each component has a weight and a set of factor vectors (one per mode).
#[derive(Debug, Clone)]
pub struct CPDecomposition {
    pub rank: usize,
    pub weights: Vec<Trit>,
    /// factors[component][mode] = factor vector for that component and mode
    pub factors: Vec<Vec<Vec<Trit>>>,
}

/// Simplified CP decomposition for ternary tensors.
///
/// This is a sign-based decomposition: for each rank-1 component, the factor
/// vectors are derived from the sign pattern of the tensor along each mode.
/// A very simplified approach for demonstration.
pub fn cp_decompose(tensor: &TernaryTensor, rank: usize) -> CPDecomposition {
    let num_modes = tensor.shape.len();
    let mut weights = vec![Trit::Zero; rank];
    let mut factors = Vec::new();

    for r in 0..rank {
        let mut mode_factors = Vec::new();
        for m in 0..num_modes {
            let dim = tensor.shape[m];
            // Sum along all modes except m to get a factor vector
            let mut factor = vec![Trit::Zero; dim];
            for i in 0..dim {
                let mut sum: i32 = 0;
                // Sum all elements where index[m] == i
                let mut index = vec![0usize; num_modes];
                for flat in 0..tensor.data.len() {
                    let mut remaining = flat;
                    for d in (0..num_modes).rev() {
                        index[d] = remaining % tensor.shape[d];
                        remaining /= tensor.shape[d];
                    }
                    if index[m] == i {
                        sum += tensor.data[flat].to_i8() as i32;
                    }
                }
                factor[i] = Trit::from_i8(sum.clamp(-1, 1) as i8).unwrap_or(Trit::Zero);
            }
            mode_factors.push(factor);
        }

        // Weight: dominant sign of all non-zero elements
        let pos_count = tensor.data.iter().filter(|&&t| t == Trit::Pos).count();
        let neg_count = tensor.data.iter().filter(|&&t| t == Trit::Neg).count();
        weights[r] = if pos_count > neg_count {
            Trit::Pos
        } else if neg_count > pos_count {
            Trit::Neg
        } else {
            Trit::Zero
        };

        factors.push(mode_factors);
    }

    CPDecomposition { rank, weights, factors }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = TernaryTensor::zeros(vec![2, 3]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.len(), 6);
        assert_eq!(t.shape, vec![2, 3]);
        assert!(t.data.iter().all(|&v| v == Trit::Zero));
    }

    #[test]
    fn test_tensor_get_set() {
        let mut t = TernaryTensor::zeros(vec![2, 3]);
        t.set(&[1, 2], Trit::Pos);
        assert_eq!(t.get(&[1, 2]), Trit::Pos);
        assert_eq!(t.get(&[0, 0]), Trit::Zero);
    }

    #[test]
    fn test_tensor_from_vec() {
        let t = TernaryTensor::from_vec(
            vec![2, 2],
            vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos],
        );
        assert_eq!(t.get(&[0, 0]), Trit::Pos);
        assert_eq!(t.get(&[0, 1]), Trit::Neg);
        assert_eq!(t.get(&[1, 0]), Trit::Zero);
        assert_eq!(t.get(&[1, 1]), Trit::Pos);
    }

    #[test]
    fn test_tensor_elementwise_add() {
        let a = TernaryTensor::from_vec(vec![2], vec![Trit::Pos, Trit::Neg]);
        let b = TernaryTensor::from_vec(vec![2], vec![Trit::Pos, Trit::Neg]);
        let c = a.add(&b);
        assert_eq!(c.data[0], Trit::Pos); // 1+1 = 2, clamped to 1
        assert_eq!(c.data[1], Trit::Neg); // -1 + -1 = -2, clamped to -1
    }

    #[test]
    fn test_tensor_multiply_elementwise() {
        let a = TernaryTensor::from_vec(vec![2], vec![Trit::Pos, Trit::Neg]);
        let b = TernaryTensor::from_vec(vec![2], vec![Trit::Neg, Trit::Pos]);
        let c = a.multiply(&b);
        assert_eq!(c.data[0], Trit::Neg);
        assert_eq!(c.data[1], Trit::Neg);
    }

    #[test]
    fn test_tensor_negate() {
        let t = TernaryTensor::from_vec(vec![3], vec![Trit::Pos, Trit::Zero, Trit::Neg]);
        let n = t.negate();
        assert_eq!(n.data[0], Trit::Neg);
        assert_eq!(n.data[1], Trit::Zero);
        assert_eq!(n.data[2], Trit::Pos);
    }

    #[test]
    fn test_matmul() {
        let a = TernaryTensor::matrix(2, 2, vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Pos]);
        let b = TernaryTensor::matrix(2, 2, vec![Trit::Pos, Trit::Pos, Trit::Zero, Trit::Zero]);
        let c = matmul(&a, &b);
        // [1 0; 0 1] * [1 1; 0 0] = [1 1; 0 0]
        assert_eq!(c.get(&[0, 0]), Trit::Pos);
        assert_eq!(c.get(&[0, 1]), Trit::Pos);
        assert_eq!(c.get(&[1, 0]), Trit::Zero);
        assert_eq!(c.get(&[1, 1]), Trit::Zero);
    }

    #[test]
    fn test_chain_multiply() {
        let a = TernaryTensor::matrix(2, 2, vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Pos]);
        let b = TernaryTensor::matrix(2, 2, vec![Trit::Neg, Trit::Zero, Trit::Zero, Trit::Neg]);
        let c = chain_multiply(&[a, b]);
        // I * (-I) = -I
        assert_eq!(c.get(&[0, 0]), Trit::Neg);
        assert_eq!(c.get(&[1, 1]), Trit::Neg);
        assert_eq!(c.get(&[0, 1]), Trit::Zero);
    }

    #[test]
    fn test_broadcast() {
        let a = TernaryTensor::from_vec(vec![1, 3], vec![Trit::Pos, Trit::Neg, Trit::Zero]);
        let b = a.broadcast(&[2, 3]);
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.get(&[0, 0]), Trit::Pos);
        assert_eq!(b.get(&[1, 0]), Trit::Pos);
        assert_eq!(b.get(&[0, 1]), Trit::Neg);
    }

    #[test]
    fn test_contract() {
        let t = TernaryTensor::from_vec(
            vec![2, 2],
            vec![Trit::Pos, Trit::Neg, Trit::Pos, Trit::Neg],
        );
        let c = contract(&t, &[0]);
        // Sum along axis 0: [Pos+Pos, Neg+Neg] = [Pos, Neg] (clamped)
        assert_eq!(c.shape, vec![2]);
        assert_eq!(c.data[0], Trit::Pos);
        assert_eq!(c.data[1], Trit::Neg);
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = TernaryTensor::from_vec(vec![2, 2], vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Neg]);
        let sparse = SparseTernaryTensor::from_dense(&dense);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(&[0, 0]), Trit::Pos);
        assert_eq!(sparse.get(&[1, 1]), Trit::Neg);
        assert_eq!(sparse.get(&[0, 1]), Trit::Zero);
    }

    #[test]
    fn test_sparse_to_dense() {
        let mut sparse = SparseTernaryTensor::new(vec![2, 2]);
        sparse.set(vec![0, 0], Trit::Pos);
        sparse.set(vec![1, 1], Trit::Neg);
        let dense = sparse.to_dense();
        assert_eq!(dense.get(&[0, 0]), Trit::Pos);
        assert_eq!(dense.get(&[1, 1]), Trit::Neg);
        assert_eq!(dense.get(&[0, 1]), Trit::Zero);
    }

    #[test]
    fn test_sparse_add() {
        let mut a = SparseTernaryTensor::new(vec![2]);
        a.set(vec![0], Trit::Pos);
        let mut b = SparseTernaryTensor::new(vec![2]);
        b.set(vec![0], Trit::Neg);
        let c = a.add(&b);
        assert_eq!(c.get(&[0]), Trit::Zero);
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_sparse_density() {
        let mut s = SparseTernaryTensor::new(vec![4, 4]);
        s.set(vec![0, 0], Trit::Pos);
        s.set(vec![1, 1], Trit::Neg);
        assert!((s.density() - 2.0 / 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_trit_multiply() {
        assert_eq!(Trit::Pos.multiply(Trit::Neg), Trit::Neg);
        assert_eq!(Trit::Neg.multiply(Trit::Neg), Trit::Pos);
        assert_eq!(Trit::Zero.multiply(Trit::Pos), Trit::Zero);
    }

    #[test]
    fn test_trit_add() {
        assert_eq!(Trit::Pos.add(Trit::Pos), Trit::Pos);
        assert_eq!(Trit::Neg.add(Trit::Neg), Trit::Neg);
        assert_eq!(Trit::Pos.add(Trit::Neg), Trit::Zero);
    }

    #[test]
    fn test_tensor_sum() {
        let t = TernaryTensor::from_vec(vec![3], vec![Trit::Pos, Trit::Zero, Trit::Neg]);
        assert_eq!(t.sum(), 0);
    }

    #[test]
    fn test_tensor_count_nonzero() {
        let t = TernaryTensor::from_vec(vec![3], vec![Trit::Pos, Trit::Zero, Trit::Neg]);
        assert_eq!(t.count_nonzero(), 2);
    }

    #[test]
    fn test_cp_decompose() {
        let t = TernaryTensor::from_vec(vec![2, 2], vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Pos]);
        let cp = cp_decompose(&t, 2);
        assert_eq!(cp.rank, 2);
        assert_eq!(cp.weights.len(), 2);
        assert_eq!(cp.factors.len(), 2);
        assert_eq!(cp.factors[0].len(), 2); // 2 modes
    }

    #[test]
    fn test_tensor_filled() {
        let t = TernaryTensor::filled(vec![2, 3], Trit::Pos);
        assert!(t.data.iter().all(|&v| v == Trit::Pos));
    }

    #[test]
    fn test_matrix_identity() {
        let i = TernaryTensor::matrix(2, 2, vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Pos]);
        let result = matmul(&i, &i);
        assert_eq!(result.get(&[0, 0]), Trit::Pos);
        assert_eq!(result.get(&[1, 1]), Trit::Pos);
        assert_eq!(result.get(&[0, 1]), Trit::Zero);
    }
}
