import numpy as np

def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference implementation for 1D convolution using NumPy.
    
    Args:
        a: Input array
        b: Kernel array
        
    Returns:
        Result of 1D convolution
    """
    return np.convolve(a, b)
