from Tensor import Tensor

class DenseMatrix(Tensor):
    def __init__(self, matrix_data):
        if not isinstance(matrix_data, list):
            matrix_data = [[x] for x in matrix_data]  # Convert to nested list
        elif not isinstance(matrix_data[0], list):
            matrix_data = [matrix_data]  # Convert to nested list
        self.data = matrix_data
        self.shape = self._get_shape(matrix_data)
    def __add__(self, other):
        return DenseMatrix(self._element_wise_operation(other, lambda x, y: x+y))
    def __sub__(self, other):
        return DenseMatrix(self._element_wise_operation(other, lambda x, y: x-y))
    def __mul__(self, other):
        return DenseMatrix(self._element_wise_operation(other, lambda x, y: x*y))
    def __matmul__(self, other):
        if not isinstance(other, DenseMatrix):
            raise ValueError("Other operand must be a Matrix instance")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication")
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
        
    
  