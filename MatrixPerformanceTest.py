import unittest
import time
import matplotlib.pyplot as plt
import scipy
from DenseMatrix import DenseMatrix
from SparseMatrix import SparseMatrixCOO
from scipy.sparse import rand as sparse_rand, diags
from scipy.sparse.linalg import aslinearoperator
import cProfile
import pstats
import os
import numpy as np

sizes = [50, 100, 200, 400, 800, 1600]  # Example sizes
class MatrixPerformanceTest(unittest.TestCase):
    def execute_and_profile(self, op_name, operation, matrix_instance):
        # Measure execution time
        operation(matrix_instance)

        # Profile the operation
        # profiler = cProfile.Profile()
        # profiler.enable()

        start_time = time.time()
        operation(matrix_instance)
        execution_time = time.time() - start_time

        # profiler.disable()
        
        # ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
        print(f'\nProfiling for {op_name}:')
        # ps.print_stats(10)  # Print stats for the top 10 functions

        return execution_time
    
    @staticmethod
    def plot_performance(sizes, results, title='Performance Comparison', ylabel='Execution Time (seconds)'):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()  # Get the current Axes instance on the current figure

        for op, data in results.items():
            for matrix_type, times in data.items():
                label = f'{matrix_type} Matrix {op.capitalize()}'
                plt.plot(sizes, times, label=label, marker='o' if matrix_type == 'Dense' else 's')

        plt.xlabel('Matrix Size')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")  # Enable grid lines for both major and minor ticks

        # Set logarithmic scale for the y-axis
        ax.set_yscale('log')

        # Optional: Set logarithmic scale for the x-axis if needed
        # ax.set_xscale('log')

        # Create 'plots' directory if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Sanitize the title to create a valid filename
        filename = title.replace(' ', '_').replace(':', '').lower() + '.png'
        filepath = os.path.join(plots_dir, filename)

        # Save the plot to the file
        plt.savefig(filepath)
        print(f'Plot saved as {filepath}')

        # Optionally, display the plot
        # plt.show()

    def execute_operations(self, sizes, operations, results):
        for size in sizes:
            shape = (size, size)
            dense_matrix = DenseMatrix.random(shape)
            sparse_matrix = SparseMatrixCOO.random(shape, density=0.01)

            for op_name, op_func in operations.items():
                # Execute the operation and measure the time for DenseMatrix
                dense_time = self.execute_and_profile(op_name, op_func, dense_matrix)
                results[op_name]['Dense'].append(dense_time)

                # Execute the operation and measure the time for SparseMatrixCOO
                sparse_time = self.execute_and_profile(op_name, op_func, sparse_matrix)
                results[op_name]['Sparse'].append(sparse_time)
        return results
    def generate_solvable_matrix(self, size, regularization_factor=1e-5):
        # Ensure the matrix is non-singular by starting with an identity matrix and adding random noise
        A = np.eye(size) + np.random.rand(size, size) * regularization_factor
        # Adding a small value to diagonal elements to ensure the matrix is well-conditioned
        np.fill_diagonal(A, A.diagonal() + regularization_factor)
        
        #convert to our DenseMatrix
        A = DenseMatrix(A)
        # Generate a random vector b
        b = DenseMatrix(np.random.rand(size, 1))

        return A, b
    def generate_solvable_sparse_matrix(self, size, density=0.01, regularization_factor=1e-5):
        # Create a random sparse matrix
        A = sparse_rand(size, size, density=density, format='coo')
        
        for i in range(size):
            A.row = np.append(A.row, i)  # Add row index for diagonal element
            A.col = np.append(A.col, i)  # Add column index for diagonal element
            A.data = np.append(A.data, regularization_factor)  # Add regularization value

        # Create an instance of the SparseMatrixCOO class with the updated COO data
        A = SparseMatrixCOO(A.row, A.col, A.data, A.shape)
        

        # Generate a random vector b
        b = SparseMatrixCOO.random((size, 1), density=density)

        return A, b
    def test_basic_operations(self):
        """Tests and plots basic operations (addition, multiplication, subtraction) on DenseMatrix and SparseMatrixCOO."""
        operations = {
            'addition': lambda x: x + x,
            'multiplication': lambda x: x @ x,
            'subtraction': lambda x: x - x,
        }
        results = {op: {'Dense': [], 'Sparse': []} for op in operations.keys()}

        # Execute the operations and measure the performance
        results = self.execute_operations(sizes, operations, results)

        # Plot the performance using the plot_performance method
        for op_name in operations.keys():
            self.plot_performance(
                sizes, results,
                title=f'{op_name.capitalize()} Operation Performance Comparison'
            )

    def test_norm_operations(self):
        """Tests and plots norm operations (L1, L2) on DenseMatrix and SparseMatrixCOO."""
        operations = {
            'L1': lambda x: x.L1_norm(),
            'L2': lambda x: x.L2_norm(),
            'max_norm': lambda x: x.max_norm(),
        }
        results = {op: {'Dense': [], 'Sparse': []} for op in operations.keys()}
        results = self.execute_operations(sizes, operations, results)
        # Plot the performance using the plot_performance method
        for op_name in operations.keys():
            self.plot_performance(
                sizes, results,
                title=f'{op_name.capitalize()} Norm Performance Comparison'
            )

    def test_svd_operations(self):
        """Tests and plots SVD operations on DenseMatrix and SparseMatrixCOO."""
        operations = {
            'svd': lambda x: x.svd(),
        }
        results = {op: {'Dense': [], 'Sparse': []} for op in operations.keys()}
        results = self.execute_operations(sizes, operations, results)
        # Plot the performance using the plot_performance method
        for op_name in operations.keys():
            self.plot_performance(
                sizes, results,
                title=f'{op_name.capitalize()} Performance Comparison'
            )
    def test_eigenvalue_operations(self):
        """Tests and plots eigenvalue operations on DenseMatrix and SparseMatrixCOO."""
        operations = {
            'compute_eigenvalues': lambda x: x.compute_eigenvalues(),
        }
        results = {op: {'Dense': [], 'Sparse': []} for op in operations.keys()}
        results = self.execute_operations(sizes, operations, results)
        # Plot the performance using the plot_performance method
        for op_name in operations.keys():
            self.plot_performance(
                sizes, results,
                title=f'{op_name.capitalize()} Performance Comparison'
            )

    def test_solve_operations(self):
        # Initialize results with the required structure
        results = {
            'Solve': {  # 'Solve' operation
                'Dense': [],  # Empty list for dense matrix times
                'Sparse': []  # Empty list for sparse matrix times
            }
        }

        for size in sizes:
            # Generate a solvable dense matrix A and vector b for each size
            A_dense, b_dense = self.generate_solvable_matrix(size)
            # Generate a solvable sparse matrix A and vector b for each size
            A_sparse, b_sparse = self.generate_solvable_sparse_matrix(size)

            # Solve using the DenseMatrix and measure the time
            start_time = time.time()
            A_dense.solve(b_dense)
            dense_time = time.time() - start_time
            results['Solve']['Dense'].append(dense_time)  # Append dense solve time

            # Solve using the SparseMatrixCOO and measure the time
            start_time = time.time()
            A_sparse.solve(b_sparse)
            sparse_time = time.time() - start_time
            results['Solve']['Sparse'].append(sparse_time)  # Append sparse solve time

        # Plot the performance comparison
        self.plot_performance(sizes, results, title='Solve Operation Performance Comparison', ylabel='Execution Time (seconds)')

if __name__ == '__main__':
    unittest.main()
