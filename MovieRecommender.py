from DenseMatrix import DenseMatrix
from SparseMatrix import SparseMatrixCOO as Sp
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
# Class for recommending movies to users
"""
1. Create a dense matrix representing the user-movie ratings
2. Convert to binary and sparse matrix as a result
3. perform SVD on the matrix
4. Run implemented recommendation algorithm
""" 

def create_movie_data():
    #Classifier that thresholds the data
    threshold = 0.7

    dense_movie_data = DenseMatrix.random(1000, 10, 0, 5)
    binary_movie_data = dense_movie_data.convert_to_binary(threshold)

    #perform svd with scipy
    U, S, V = sp.svd(dense_movie_data.data)

    #plot singular values
    plt.plot(S)
    plt.title("Singular Values of the Movie Data")
    plt.show()




    

    