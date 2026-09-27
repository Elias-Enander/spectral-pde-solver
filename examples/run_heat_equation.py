import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from spectral_pde.core import smooth_bump_function
from spectral_pde.solvers import rk4_fourier_step

def main():
    print("Simulating Transient Heat Equation via Pseudo-Spectral Method...")
    
    # System Parameters
    D = 1.0
    N = 64
    T_end = 0.1

    # Calculate stable time-step dynamically to prevent numerical blow-up
    dt_max = 2.8 / (D * np.pi**2 * N**2)
    dt = 0.9 * dt_max  # Use 90% of the theoretical limit for safety
    
    x = np.arange(N) / N
    u_hat = np.zeros(N, dtype=complex)  # Initial condition f(x) = 0
    
    # Source term evaluation in Fourier space
    def g_hat_eval(t):
        y = 2.0 * np.abs(x - 0.5)
        g = 100.0 * smooth_bump_function(y, delta=0.05) * np.exp(-100.0 * t)
        return np.fft.fftshift(np.fft.fft(g)) / N
    
    num_steps = int(np.round(T_end / dt))
    num_saves = 25
    save_interval = num_steps // num_saves
    
    U_history = np.zeros((num_saves, N))
    t_history = np.linspace(0, T_end, num_saves)
    
    t = 0.0
    save_idx = 0
    
    for step in range(1, num_steps + 1):
        u_hat = rk4_fourier_step(u_hat, t, dt, N, D, g_hat_eval)
        
        if step % save_interval == 0 and save_idx < num_saves:
            U_history[save_idx, :] = (N * np.fft.ifft(np.fft.ifftshift(u_hat))).real
            save_idx += 1
            
        t += dt

    # Waterfall Plot (3D Wireframe)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    X, T_mesh = np.meshgrid(x, t_history)
    ax.plot_wireframe(X, T_mesh, U_history, rstride=1, cstride=0, color='b', alpha=0.7)
    
    ax.set_xlabel("Spatial Domain $x$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Temperature $u(x,t)$")
    ax.set_title("Transient PDE Evolution (Waterfall Plot)")
    
    plt.tight_layout()
    plt.savefig("waterfall_plot.png", dpi=300)
    print("Saved 'waterfall_plot.png'")

if __name__ == "__main__":
    main()