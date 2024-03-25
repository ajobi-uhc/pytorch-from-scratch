from DenseMatrix import DenseMatrix
from SparseMatrix import SparseMatrixCOO as Sp
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
# Class for recommending movies to users
"""
Chose a threshold of 0.3, so most people generally like the movies to simulate a real world scenario
""" 

#Based on the singular values, we can see that the singular values decay rapidly, so we can truncate the SVD
def truncate_svd(U, S, V, k):
    U_trunc = U[:, :k]
    S_trunc = S[:k]
    V_trunc = V[:k, :]
    return U_trunc, S_trunc, V_trunc

def recommend(liked_movie_index, V, amount_to_recommend):
    recommended = []
    for i in range(len(V)):
        if i != liked_movie_index:
            similarity = np.dot(V[liked_movie_index], V[i])
            recommended.append((i, similarity))
    recommended.sort(key=lambda x: x[1], reverse=True)
    return recommended[:amount_to_recommend]

def main():
    shape = (1000, 50)
    dense_movie_data = DenseMatrix.random(shape)
    threshold = 0.7
    binary_movie_data = dense_movie_data.transform_into_binary(threshold)
    U, S, V = sp.svd(binary_movie_data.data)


    #plot values of S with respect to index in array S one ordered by magnitude
    #order the singular values
    S = np.sort(S)[::-1]
    print("Singular values: ", S)

    plt.plot(S)
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular values of movie data")
    plt.show()


    #truncate the SVD
    k = 8
    U_trunc, S_trunc, V_trunc = truncate_svd(U, S, V, k)

    #order the singular values
    S_trunc = np.sort(S_trunc)[::-1]
    print("Truncated singular values: ", S_trunc)

    V = V_trunc

    liked_movie_index = 0
    amount_to_recommend = 5
    recommended_movies = recommend(liked_movie_index, V, amount_to_recommend)
    print("Recommended movies: ", recommended_movies)    

main()