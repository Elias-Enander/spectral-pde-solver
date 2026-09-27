"""
Numerical time-integration solvers for Partial Differential Equations (PDEs) in Fourier space.
"""
import numpy as np

def rk4_fourier_step(u_hat: np.ndarray, t: float, dt: float, N: int, D: float, g_hat_func) -> np.ndarray:
    """
    Advances the PDE state u_hat by one dt using 4th-order Runge-Kutta.
    Equation: u_t = D * u_xx + g(x,t)
    """
    k_vec = np.arange(-N/2, N/2)
    k2 = -(2.0 * np.pi * k_vec)**2
    
    # RK4 Step 1
    g_hat1 = g_hat_func(t)
    h1 = D * k2 * u_hat + g_hat1
    
    # RK4 Step 2
    g_hat2 = g_hat_func(t + 0.5 * dt)
    h2 = D * k2 * (u_hat + 0.5 * dt * h1) + g_hat2
    
    # RK4 Step 3
    h3 = D * k2 * (u_hat + 0.5 * dt * h2) + g_hat2
    
    # RK4 Step 4
    g_hat4 = g_hat_func(t + dt)
    h4 = D * k2 * (u_hat + dt * h3) + g_hat4
    
    # Update state
    u_hat_next = u_hat + (dt / 6.0) * (h1 + 2.0 * h2 + 2.0 * h3 + h4)
    return u_hat_next