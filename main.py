from DenseMatrix import DenseMatrix
from SparseMatrix import SparseMatrixCOO
def run():
    data_a = [1,3,4]
    data_b = [[1,2,1],[4,5,1],[1,2,1], [3,4,1]]
    a = DenseMatrix(data_a)
    b = DenseMatrix(data_b)
    print("original shape A", a.shape)
    print("original shape B", b.shape)
    c = a + b
    print(c)

    c = a - b
    print("sub", c)

    x = DenseMatrix([[1,2], [1,2]])
    y = DenseMatrix([[1,2], [1,2]])

    z = x @ y
    print("mat mul", z)

    z = a * b
    print("elementwise mul", z)

    sp1 = [[0,0,0],[0,2,0],[0,1,0]]
    sp2 = [[0,0,3],[3,2,0],[0,1,0]]

    new_sparse = SparseMatrixCOO([1, 2], [1,1], [2, 1], (3,3))
    print(new_sparse)
    

run()