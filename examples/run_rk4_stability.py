"""
Numerical stability analysis of the RK4 time-integration scheme in Fourier space.
Verifies the theoretical stability boundary dt_max = 2.8 / (D * pi^2 * N^2).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from spectral_pde.solvers import rk4_fourier_step

def main():
    print("Evaluating RK4 Stability Limit for Spectral PDE Solver...")

    # System parameters[cite: 24]
    D = 1.0
    b = 6.0
    T_end = 0.1
    N = 20

    # Theoretical stability limit for RK4 with Spectral 2nd Derivative[cite: 24]
    # The stability region of RK4 on the real axis extends to approx -2.8[cite: 24]
    dt_max = 2.8 / (D * np.pi**2 * N**2)
    print(f"Theoretical dt_max for N={N}: {dt_max:.4e} s")

    x = np.arange(N) / N

    # Initial condition and Source term[cite: 24]
    def get_initial_u_hat():
        f = np.cos(b * np.pi * x)
        return np.fft.fftshift(np.fft.fft(f)) / N

    def g_hat_eval(t):
        g = (b**2 * np.pi**2 - 1.0) * np.cos(b * np.pi * x) * np.exp(-t)
        return np.fft.fftshift(np.fft.fft(g)) / N

    # ---------------------------------------------------------
    # TEST 1: Stable integration (dt = 0.95 * dt_max)
    # ---------------------------------------------------------
    dt_stable = 0.95 * dt_max
    u_hat_stable = get_initial_u_hat()
    t = 0.0
    while t < T_end - 1e-9:
        dt_step = min(dt_stable, T_end - t)
        u_hat_stable = rk4_fourier_step(u_hat_stable, t, dt_step, N, D, g_hat_eval)
        t += dt_step
    
    u_stable = (N * np.fft.ifft(np.fft.ifftshift(u_hat_stable))).real

    # ---------------------------------------------------------
    # TEST 2: Unstable integration (dt = 1.05 * dt_max)
    # ---------------------------------------------------------
    dt_unstable = 1.05 * dt_max
    u_hat_unstable = get_initial_u_hat()
    t = 0.0
    
    # We run fewer steps here because it will rapidly blow up towards infinity (Overflow)
    try:
        while t < T_end - 1e-9:
            dt_step = min(dt_unstable, T_end - t)
            u_hat_unstable = rk4_fourier_step(u_hat_unstable, t, dt_step, N, D, g_hat_eval)
            t += dt_step
        u_unstable = (N * np.fft.ifft(np.fft.ifftshift(u_hat_unstable))).real
    except OverflowError:
        u_unstable = np.full_like(x, np.nan) # Handle math overflow gracefully for the plot

    # Exact analytical solution[cite: 24]
    u_exact = np.cos(b * np.pi * x) * np.exp(-T_end)

    # ---------------------------------------------------------
    # Generate Verification Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # Plot 1: Stable
    plt.subplot(1, 2, 1)
    plt.plot(x, u_exact, 'k-', linewidth=4, alpha=0.3, label='Exact Solution')
    plt.plot(x, u_stable, 'b-o', linewidth=1.5, markersize=5, label=rf'Stable ($\Delta t = 0.95 \Delta t_{{max}}$)')
    plt.title("Stable Integration", fontsize=12, fontweight="bold")
    plt.xlabel("Spatial Domain $x$")
    plt.ylabel(f"Temperature $u(x, t={T_end})$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Plot 2: Unstable
    plt.subplot(1, 2, 2)
    plt.plot(x, u_exact, 'k-', linewidth=4, alpha=0.3, label='Exact Solution')
    plt.plot(x, u_unstable, 'r-s', linewidth=1.5, markersize=5, label=rf'Unstable ($\Delta t = 1.05 \Delta t_{{max}}$)')
    plt.title("Unstable Integration (Numerical Blow-up)", fontsize=12, fontweight="bold")
    plt.xlabel("Spatial Domain $x$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("rk4_stability.png", dpi=300)
    print("Saved 'rk4_stability.png'")

if __name__ == "__main__":
    main()