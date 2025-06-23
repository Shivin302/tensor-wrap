import numpy as np
import scipy.signal as signal

def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference implementation for 2D convolution using NumPy.
    
    Args:
        a: Input array
        b: Kernel array
        
    Returns:
        Result of 2D convolution
    """
    return signal.convolve2d(a, b, mode='full')
