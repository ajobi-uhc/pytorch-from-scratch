class Tensor:
    def __init__(self):
        raise NotImplementedError("This class is not meant to be instantiated directly.")
    def __str__(self):
        return str(self.data)
    def _get_shape(self, matrix):
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            matrix = matrix[0] if matrix else []  # Handle empty lists correctly
        return tuple(shape)
    def is_dense(self, X):
        return hasattr(X, '_dense_identifier')
    def is_sparse(self, X):
        return hasattr(X, '_sparse_identifier')
    def _broadcast_shape(self, shape_a, shape_b):
        """Calculate the broadcasted shape of two shapes."""
        # Convert shapes to tuples to ensure consistency
        shape_a = tuple(shape_a)
        shape_b = tuple(shape_b)
        # Initialize the result shape as a list
        result_shape = []
        # Pad the shorter shape with 1s for alignment
        if len(shape_a) > len(shape_b):
            shape_b = (1,) * (len(shape_a) - len(shape_b)) + shape_b
        elif len(shape_b) > len(shape_a):
            shape_a = (1,) * (len(shape_b) - len(shape_a)) + shape_a
        # Calculate the broadcasted shape
        for dim_a, dim_b in zip(shape_a, shape_b):
            if dim_a == 1 or dim_b == 1 or dim_a == dim_b:
                result_shape.append(max(dim_a, dim_b))
            else:
                raise ValueError("Shapes cannot be broadcast together: {} and {}".format(shape_a, shape_b))
        # Convert the result to a tuple before returning
        return tuple(result_shape)

    def _broadcast_to(self, matrix, target_shape):
        def expand(matrix, shape, target):
            if len(shape) == 0:
                return matrix
            expanded = [expand(matrix[i % len(matrix)], shape[1:], target[1:]) for i in range(target[0])]
            return expanded

        # Ensure the base case of scalar expansion is handled
        if not isinstance(matrix, list):
            matrix = [[matrix]]

        current_shape = self._get_shape(matrix)
        return expand(matrix, current_shape, target_shape)
    
    def apply_op(self, a, b, op):
        if isinstance(a, list) and isinstance(b, list):
            return [self.apply_op(ai, bi, op) for ai, bi in zip(a, b)]
        elif isinstance(a, list):
            return [self.apply_op(ai, b, op) for ai in a]
        elif isinstance(b, list):
            return [self.apply_op(a, bi, op) for bi in b]
        else:
            return op(a, b)

    def _element_wise_operation(self, other, op):
        if not isinstance(other, Tensor):
            raise ValueError("Other operand must be a Matrix instance")
        #check if broadcasting is needed
        if self.shape != other.shape:
            result_shape = self._broadcast_shape(self.shape, other.shape)
            broadcasted_self = self._broadcast_to(self.data, result_shape)
            broadcasted_other = other._broadcast_to(other.data, result_shape)
            result_data = self.apply_op(broadcasted_self, broadcasted_other, op)
        else:
            result_data = self.apply_op(self.data, other.data, op)        
        return result_data