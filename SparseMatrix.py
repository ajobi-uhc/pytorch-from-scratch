from Tensor import Tensor
class SparseMatrixCOO(Tensor):
    def __init__(self, rows, cols, values, shape):
        # Validate lengths of rows, cols, and values arrays
        if not (len(rows) == len(cols) == len(values)):
            raise ValueError("Rows, columns, and values arrays must have the same length")
        data = {}
        data["rows"] = rows
        data["cols"] = cols
        data["values"] = values
        self.data = data
        self.shape = shape  # Shape of the sparse matrix (tuple)

    @staticmethod
    def from_dict(matrix_data, shape):
        rows, cols, values = [], [], []
        for (row, col), value in matrix_data.items():
            rows.append(row)
            cols.append(col)
            values.append(value)
        return SparseMatrixCOO(rows, cols, values, shape)

    def to_dict(self):
        matrix_data = {}
        for row, col, value in zip(self.data.rows, self.data.cols, self.data.values):
            matrix_data[(row, col)] = value
        return matrix_data

    def __add__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")
        rows = self.data.rows + other.data.rows
        cols = self.data.cols + other.data.cols
        values = self.data.values + other.data.values
        return SparseMatrixCOO(rows, cols, values, self.shape)
    
    def __sub__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")
        rows = self.data.rows + other.data.rows
        cols = self.data.cols + other.data.cols
        values = self.data.values + [-val for val in other.data.values]
        return SparseMatrixCOO(rows, cols, values, self.shape)
    
    def __mul__(self, other):
        # Element-wise multiplication
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape != other.shape:
            raise ValueError("Shapes of the matrices must match")
        result = {}
        
        return SparseMatrixCOO.from_dict(result, self.shape)
    
    def __matmul__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Other operand must be a SparseMatrixCOO instance")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication")
        result = {}

        return SparseMatrixCOO.from_dict(result, (self.shape[0], other.shape[1]))

