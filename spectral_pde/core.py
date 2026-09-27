"""
Core mathematical functions and exact analytical definitions.
"""
import numpy as np

def smooth_bump_function(y: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """
    Computes the C^infinity smooth bump function eta(y, delta).
    Used to generate smooth step-like transitions in PDE source terms.
    """
    alpha = 1.0 - 2.0 * delta
    val = np.zeros_like(y)
    
    def h(s):
        # Suppress divide-by-zero warnings for boundary limits
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.exp((2.0 * np.exp(-1.0 / s)) / (s - 1.0))
            result[np.isnan(result)] = 0.0
            return result

    for i, yi in enumerate(y):
        if yi <= delta:
            val[i] = 1.0
        elif yi >= 1.0 - delta:
            val[i] = 0.0
        else:
            s_val = (yi - delta) / alpha
            h_s = h(np.array([s_val]))[0]
            h_1ms = h(np.array([1.0 - s_val]))[0]
            val[i] = h_s / (h_s + h_1ms)
            
    return val

def gaussian_pulse(x: np.ndarray, alpha: float = 400.0) -> np.ndarray:
    """Analytical Gaussian pulse for differentiation tests."""
    return np.exp(-alpha * (x - np.pi / 5.0)**2)

def exact_gaussian_derivatives(x: np.ndarray, alpha: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns the exact analytical 1st and 2nd derivatives of the Gaussian pulse."""
    f = gaussian_pulse(x, alpha)
    f_prime = -2.0 * alpha * (x - np.pi / 5.0) * f
    f_double_prime = (-2.0 * alpha + 4.0 * alpha**2 * (x - np.pi / 5.0)**2) * f
    return f_prime, f_double_prime