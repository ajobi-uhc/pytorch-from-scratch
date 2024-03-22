from Tensor import Tensor
import scipy.linalg as sp
import numpy as np
class DenseMatrix(Tensor):
    _dense_identifier = True
    def __init__(self, matrix_data):
        # Initialize self.shape to a default value
        self.shape = (0, 0)  # Default shape

        if isinstance(matrix_data, np.ndarray):
            if len(matrix_data.shape) == 2:  # Ensure matrix_data is 2D
                matrix_data = matrix_data.tolist()
                self.data = matrix_data
                self.shape = self._get_shape(matrix_data)
            else:
                raise ValueError("matrix_data must be a 2D numpy ndarray")
        elif isinstance(matrix_data, list) and all(isinstance(row, list) for row in matrix_data):
            self.data = matrix_data
            self.shape = self._get_shape(matrix_data)
        else:
            raise ValueError("matrix_data must be a 2D list or a 2D numpy ndarray")
    def __str__(self) -> str:
        return super().__str__()
    def __add__(self, other):
        return DenseMatrix(self._element_wise_operation(other, lambda x, y: x+y))
    def __sub__(self, other):
        return DenseMatrix(self._element_wise_operation(other, lambda x, y: x-y))
    def __mul__(self, other):
        #implement elementwise multiplication if other matrix is sparse
        if self.is_sparse(other):
            result_values = []
            for row, col, value in zip(other.rows, other.cols, other.values):
                result_values.append(self.data[row][col] * value)
            #return Dense Matrix
            return DenseMatrix(result_values)
        else:
            return DenseMatrix(self._element_wise_operation(other, lambda x, y: x*y))
    def __matmul__(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication")

        #implement Sparse Matrix multiplication
        if self.is_sparse(other):
            #get copy of self matrix
            self_copy = self.data.copy()
            for row, col, value in zip(other.rows, other.cols, other.values):
                element = 0                
                current_value = self.data[col][row]
                element = current_value * value
                self_copy[col][row] = element
            return DenseMatrix(self_copy)
        else:
            result = []        
            for i in range(self.shape[0]):
                row = []
                for j in range(other.shape[1]):
                    element = 0
                    for k in range(self.shape[1]):
                        element += self.data[i][k] * other.data[k][j]
                    row.append(element)
                result.append(row)
            
            return result    

    def to_dense_matrix(self, sparse_matrix):
        matrix = [[0 for _ in range(sparse_matrix.shape[1])] for _ in range(sparse_matrix.shape[0])]
        for row, col, value in zip(sparse_matrix.rows, sparse_matrix.cols, sparse_matrix.values):
            matrix[row][col] = value
        return DenseMatrix(matrix)
    @classmethod
    def random(cls, shape, seed=None):
        if seed is not None:
            np.random.seed(seed)
        random_data = np.random.rand(*shape).tolist()
        return cls(random_data)
    def L1_norm(self):
        return sum([abs(x) for row in self.data for x in row])
    def L2_norm(self):
        return sum([x**2 for row in self.data for x in row])**0.5
    def max_norm(self):
        return max([max(row) for row in self.data])        
    def compute_eigenvalues(self):
        #implement eigenvalues computation as a square matrix
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square to compute eigenvalues")
        #use scipy to compute eigenvalues
        return sp.eigvals(self.data)
    def svd(self):
        #implement SVD computation
        return sp.svd(self.data)
    def solve(self, b):
        if self.is_sparse(b):
            b = self.to_dense_matrix(b).data
        elif isinstance(b, DenseMatrix):
            b = b.data
        #implement solving linear system of equations
        return sp.solve(self.data, b)
    def transform_into_binary(self, threshold):
        transform = lambda x: 1 if x > threshold else 0
        return DenseMatrix([[transform (x) for x in row] for row in self.data])
        

        
    
  