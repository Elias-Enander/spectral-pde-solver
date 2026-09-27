import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from spectral_pde.core import gaussian_pulse, exact_gaussian_derivatives
from spectral_pde.differentiation import SpectralDerivatives

def main():
    print("Evaluating Spectral Differentiation Convergence...")
    
    alpha = 400.0
    m_vals = [4, 5, 6, 7, 8]
    N_vals = [2**m for m in m_vals]
    
    err_prime = []
    err_double_prime = []
    
    for N in N_vals:
        x = np.arange(N) / N
        f = gaussian_pulse(x, alpha)
        
        f_prime_num, f_double_prime_num = SpectralDerivatives.compute_derivatives(f, N)
        f_prime_ex, f_double_prime_ex = exact_gaussian_derivatives(x, alpha)
        
        # Compute L2 error norms
        e1 = np.sqrt(np.sum(np.abs(f_prime_num - f_prime_ex)**2) / N)
        e2 = np.sqrt(np.sum(np.abs(f_double_prime_num - f_double_prime_ex)**2) / N)
        
        err_prime.append(e1)
        err_double_prime.append(e2)
        print(f"N = {N:3d} | L2 Error (1st Deriv): {e1:.2e} | L2 Error (2nd Deriv): {e2:.2e}")

    plt.figure(figsize=(8, 5))
    plt.loglog(N_vals, err_prime, 'b-o', linewidth=2, label="1st Derivative Error")
    plt.loglog(N_vals, err_double_prime, 'r-s', linewidth=2, label="2nd Derivative Error")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Grid Resolution $N$")
    plt.ylabel("Discrete $L_2$ Error Norm")
    plt.title("Spectral Convergence of Spatial Derivatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectral_convergence.png", dpi=300)
    print("Saved 'spectral_convergence.png'")

if __name__ == "__main__":
    main()