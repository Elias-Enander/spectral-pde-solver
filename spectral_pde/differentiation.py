"""
Spectral differentiation operators using Fast Fourier Transforms (FFT).
"""
import numpy as np

class SpectralDerivatives:
    """Computes spatial derivatives in the frequency domain using FFT/iFFT."""
    
    @staticmethod
    def get_wavenumbers(N: int) -> np.ndarray:
        """Returns the centered wavenumber vector k."""
        return np.arange(-N/2, N/2)

    @staticmethod
    def compute_derivatives(f: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 1st and 2nd spatial derivatives of f(x) via spectral methods.
        Returns (df/dx, d2f/dx2).
        """
        # Forward transform with normalization
        f_hat = np.fft.fftshift(np.fft.fft(f)) / N
        k_vec = SpectralDerivatives.get_wavenumbers(N)
        
        # Spectral differentiation in Fourier space
        f_hat_prime = (1j * 2.0 * np.pi * k_vec) * f_hat
        f_hat_double_prime = -(2.0 * np.pi * k_vec)**2 * f_hat
        
        # Inverse transform back to physical space
        f_prime = (N * np.fft.ifft(np.fft.ifftshift(f_hat_prime))).real
        f_double_prime = (N * np.fft.ifft(np.fft.ifftshift(f_hat_double_prime))).real
        
        return f_prime, f_double_prime