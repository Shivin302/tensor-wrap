import numpy as np

def matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference implementation for matrix multiplication using NumPy.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        Result of matrix multiplication
    """
    return np.matmul(a, b)
