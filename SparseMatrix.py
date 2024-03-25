from Tensor import Tensor
import scipy.sparse.linalg as sp
from scipy.sparse import coo_matrix
import numpy as np
class SparseMatrixCOO(Tensor):
    _sparse_identifier = True
    def __init__(self, rows, cols, values, shape):
        # Validate lengths of rows, cols, and values arrays
        if not (len(rows) == len(cols) == len(values)):
            raise ValueError("Rows, columns, and values arrays must have the same length")
        self.rows = rows
        self.cols = cols
        self.values = values
        self.shape = shape  # Shape of the sparse matrix (tuple)
    def from_dict(self, rows, cols, values, shape):
        #initialize with zeros
        matrix = [[0 for _ in range(shape[1])] for _ in range(shape[0])]

        for row, col, value in zip(rows, cols, values):
            matrix[row][col] = value

        return matrix
            
    def __str__(self) -> str:
        return str(self.from_dict(self.rows, self.cols, self.values, self.shape))
    
    def __add__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")

        result_dict = {}
        for row, col, value in zip(self.rows, self.cols, self.values):
            result_dict[(row, col)] = value

        for row, col, value in zip(other.rows, other.cols, other.values):
            if (row, col) in result_dict:
                result_dict[(row, col)] += value
            else:
                result_dict[(row, col)] = value

        rows, cols, values = zip(*[(key[0], key[1], val) for key, val in result_dict.items()])
        return SparseMatrixCOO(list(rows), list(cols), list(values), self.shape)

    def __sub__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")

        result_dict = {}
        for row, col, value in zip(self.rows, self.cols, self.values):
            result_dict[(row, col)] = value

        for row, col, value in zip(other.rows, other.cols, other.values):
            if (row, col) in result_dict:
                result_dict[(row, col)] -= value
            else:
                result_dict[(row, col)] = -value

        rows, cols, values = zip(*[(key[0], key[1], val) for key, val in result_dict.items()])
        return SparseMatrixCOO(list(rows), list(cols), list(values), self.shape)
    
    
    def __mul__(self, other):
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")
        Y = other
        result_values = []
        if self.is_dense(Y):
            # perform dense matrix and sparse matrix elementwise multiply
            # only multiply the values corresponding to non-zero vals in sparse matrix
            for row, col, value in zip(self.rows, self.cols, self.values):
                # other matrices
                Yvalue = Y.data[row][col]
                value = Yvalue * value
                result_values.append(value)
        elif isinstance(other, SparseMatrixCOO):
            # perform sparse matrix elementwise multiply
            for rowX, colX, valueX in zip(self.rows, self.cols, self.values):
                for rowY, colY, valueY in zip(Y.rows, Y.cols, Y.values):
                    if rowX == rowY and colX == colY:
                        result_values.append(valueX * valueY)
                    
        return SparseMatrixCOO(self.rows, self.cols, result_values, self.shape)
        
        
    def to_scipy_sparse(self):
        """Converts this SparseMatrixCOO to a scipy.sparse.coo_matrix."""
        return coo_matrix((self.values, (self.rows, self.cols)), shape=self.shape)
    
    def __matmul__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication")
        # Create a array to store the result
        result = [[0 for _ in range(other.shape[1])] for _ in range(self.shape[0])]
        # Create a dictionary to store the non-zero values in the result
        result_dict = {}
        # Iterate over the non-zero values of the first matrix
        for i, j, value in zip(self.rows, self.cols, self.values):
            # Iterate over the non-zero values of the second matrix
            for k, l, value2 in zip(other.rows, other.cols, other.values):
                # If the column of the first matrix matches the row of the second matrix
                if j == k:
                    # Multiply the values and add to the result
                    result[i][l] += value * value2
                    # Store the result in the dictionary in a way that zip works
                    result_dict[(i, l)] = result[i][l]

        #extract rows, cols and values from dict of type dict_items([((1, 0), 6), ((1, 1), 4), ((2, 0), 3), ((2, 1), 2)])
        coords, values = zip(*result_dict.items())
        rows, cols = zip(*coords)
        # Return the result as a new SparseMatrixCOO
        return SparseMatrixCOO(rows, cols, values, (self.shape[0], other.shape[1]))    
    def to_sparse_matrix(self, dense_matrix):
        rows = []
        cols = []
        values = []
        for i in range(dense_matrix.shape[0]):
            for j in range(dense_matrix.shape[1]):
                if dense_matrix.data[i][j] != 0:
                    rows.append(i)
                    cols.append(j)
                    values.append(dense_matrix.data[i][j])
        return SparseMatrixCOO(rows, cols, values, dense_matrix.shape)
    @classmethod
    def random(cls, shape, density=0.1, value_range=(0, 10), seed=None):
        if seed is not None:
            np.random.seed(seed)
        total_elements = shape[0] * shape[1]
        non_zero_elements = int(total_elements * density)

        rows = np.random.randint(0, shape[0], size=non_zero_elements)
        cols = np.random.randint(0, shape[1], size=non_zero_elements)
        values = np.random.uniform(value_range[0], value_range[1], size=non_zero_elements).tolist()
        return cls(rows.tolist(), cols.tolist(), values, shape)
    def L1_norm(self):
        return sum([abs(val) for val in self.values])
    def L2_norm(self):
        return sum([val**2 for val in self.values])**0.5
    def max_norm(self):
        return max([abs(val) for val in self.values])
    def compute_eigenvalues(self):
        scipy_sparse_matrix = self.to_scipy_sparse()
        eigenvalues, _ = sp.eigs(scipy_sparse_matrix)
        return eigenvalues
    def svd(self):
        scipy_sparse_matrix = self.to_scipy_sparse()
        return sp.svds(scipy_sparse_matrix)
    def solve(self, b):
        if self.is_sparse(b):
            b = b.to_scipy_sparse()
        elif self.is_dense(b):
            b = np.array(b.data)
        scipy_sparse_matrix = self.to_scipy_sparse()
        #convert to csr format
        scipy_sparse_matrix = scipy_sparse_matrix.tocsr()
        solution = sp.spsolve(scipy_sparse_matrix, b)
        return solution
    def transform_into_binary(self, threshold):
        transformed_indices_and_values = [
            (row, col, 1) for row, col, value in zip(self.rows, self.cols, self.values) if value > threshold
        ]

        if not transformed_indices_and_values:
            return SparseMatrixCOO([], [], [], self.shape)

        new_rows, new_cols, new_values = zip(*transformed_indices_and_values)

        return SparseMatrixCOO(list(new_rows), list(new_cols), list(new_values), self.shape)



