"""
FxLMS Algorithm Example - Python Adaptation
============================================

This is a Python adaptation of a MATLAB FxLMS example for a single-channel 
feed-forward active noise control system. Reference: "Active Noise Control 
Systems - Algorithms and DSP Implementations," by S. M. Kuo and D. R. Morgan (1996).

System Diagram:
              +-----------+                       +   
 x(k) ---+--->|   P(z)    |--yp(k)----------------> sum --+---> e(k)
         |    +-----------+                          ^-   |
         |                                           |    |
         |        \                                ys(k)  |     
         |    +-----------+          +-----------+   |    |
         +--->|   C(z)    |--yw(k)-->|   S(z)    |---+    |
         |    +-----------+          +-----------+        |
         |            \                                   |
         |             \----------------\                 |
         |                               \                |
         |    +-----------+          +-----------+        |
         +--->|   Sh(z)   |--xs(k)-->|    LMS    |<-------+
              +-----------+          +-----------+        

Where:
- P(z): Primary path (noise propagation from source to sensor)
- S(z): Secondary path (actuator to sensor)
- C(z): Controller (adaptive filter)
- Sh(z): Estimate of S(z)

Original MATLAB code by Agustinus Oey <oeyaugust@gmail.com>
Center of Noise and Vibration Control (NoViC)
Department of Mechanical Engineering
Korea Advanced Institute of Science and Technology (KAIST)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))


#%% MAIN SCRIPT

if __name__ == "__main__":

    #%% CONFIGURATION AND SETUP
    
    # Set simulation duration (number of samples)
    T = 1000
    
    # Define the unknown paths P(z) and S(z) (simulated as FIR filters)
    # P(z): Primary path - noise propagation from source to sensor
    # S(z): Secondary path - actuator to sensor
    Pw = np.array([0.01, 0.25, 0.5, 1.0, 0.5, 0.25, 0.01])  # Primary path coefficients
    Sw = Pw * 0.25  # Secondary path coefficients

    #%% PART 1: SECONDARY PATH IDENTIFICATION
    # The first task is to estimate S(z) using the LMS algorithm
    
    print("=" * 60)
    print("PART 1: Secondary Path Identification")
    print("=" * 60)
    
    # Generate white noise for identification
    np.random.seed(42)  # For reproducibility
    x_iden = np.random.randn(T)
    
    # Send the noise through the actual secondary path S(z)
    y_iden = np.convolve(x_iden, Sw, mode='full')[:T]  # Truncate to T samples
    
    # Initialize the secondary path estimate Sh(z)
    Sh_len = 16  # Filter length for Sh(z)
    Shx = np.zeros(Sh_len)  # State buffer (input samples)
    Shw = np.zeros(Sh_len)  # Filter weights/coefficients
    e_iden = np.zeros(T)    # Identification error buffer
    
    # Apply vanilla LMS algorithm for identification
    mu_iden = 0.1  # Learning rate
    
    for k in range(T):
        # Update the state buffer (shift and insert new sample)
        Shx = np.roll(Shx, 1)
        Shx[0] = x_iden[k]
        
        # Calculate output of Sh(z)
        Shy = np.dot(Shx, Shw)
        
        # Calculate identification error
        e_iden[k] = y_iden[k] - Shy
        
        # Update weights using LMS rule
        Shw = Shw + mu_iden * e_iden[k] * Shx
    
    print(f"Final identification error (last sample): {e_iden[-1]:.6f}")
    print(f"Mean squared error (last 100 samples): {np.mean(e_iden[-100:]**2):.6f}")

    #%% PART 2: ACTIVE NOISE CONTROL USING FxLMS
    
    print("\n" + "=" * 60)
    print("PART 2: Active Noise Control using FxLMS")
    print("=" * 60)
    
    # Generate the noise signal x(k)
    input = np.random.randn(T)  # Using 'input' to match experiment_01 naming
    
    # Measure the arriving noise at the sensor position (through primary path)
    # This is the "desired" signal we want to cancel
    y_primary = np.convolve(input, Pw, mode='full')[:T]
    
    # Initialize the controller C(z)
    C_len = 16  # Controller filter length
    Cx = np.zeros(C_len)   # Controller state buffer (input samples)
    Cw = np.zeros(C_len)   # Controller weights/coefficients
    
    # Initialize secondary path simulation buffer
    Sx = np.zeros(len(Sw))  # Buffer for secondary path input
    
    # Initialize filtered-x buffer for FxLMS
    Xhx = np.zeros(C_len)   # Filtered reference signal buffer
    
    # Reinitialize Shx for the control phase (reuse the identified Shw)
    Shx = np.zeros(Sh_len)
    
    # Output buffers
    e_cont = np.zeros(T)    # Control error (residual noise)
    y_control = np.zeros(T) # Control signal (anti-noise)
    
    # Apply the FxLMS algorithm
    mu_cont = 0.1  # Learning rate for controller
    
    for k in range(T):
        # Update the controller state buffer
        Cx = np.roll(Cx, 1)
        Cx[0] = input[k]
        
        # Calculate the controller output (anti-noise signal before secondary path)
        Cy = np.dot(Cx, Cw)
        
        # Propagate through secondary path (shift buffer and insert)
        Sx = np.roll(Sx, 1)
        Sx[0] = Cy
        
        # Calculate the actual anti-noise arriving at sensor (through S(z))
        y_secondary = np.dot(Sx, Sw)
        
        # Measure the residue (error signal at microphone)
        # e(k) = yp(k) - ys(k) where yp is primary noise, ys is anti-noise
        e_cont[k] = y_primary[k] - y_secondary
        
        # Store control signal for visualization
        y_control[k] = y_secondary
        
        # Update Sh(z) state buffer for filtered-x computation
        Shx = np.roll(Shx, 1)
        Shx[0] = input[k]
        
        # Calculate filtered reference signal x'(k) = Sh(z) * x(k)
        filtered_x = np.dot(Shx, Shw)
        
        # Update filtered-x buffer
        Xhx = np.roll(Xhx, 1)
        Xhx[0] = filtered_x
        
        # Update controller weights using FxLMS rule
        Cw = Cw + mu_cont * e_cont[k] * Xhx
    
    print(f"Final residual noise (last sample): {e_cont[-1]:.6f}")
    print(f"Mean squared residual (last 100 samples): {np.mean(e_cont[-100:]**2):.6f}")
    print(f"Mean squared primary noise: {np.mean(y_primary**2):.6f}")
    print(f"Noise reduction: {10*np.log10(np.mean(y_primary**2)/np.mean(e_cont[-100:]**2)):.2f} dB")

    #%% PLOTS
    
    # Plot 1: Secondary Path Identification Results
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Identification error over time
    time_axis = np.arange(T)
    axes1[0].plot(time_axis, e_iden, label='Identification error', alpha=0.8)
    axes1[0].set_ylabel('Amplitude')
    axes1[0].set_xlabel('Discrete time k')
    axes1[0].set_title('Secondary Path Identification Error')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Subplot 2: Comparison of S(z) and Sh(z) coefficients
    tap_indices_Sw = np.arange(len(Sw))
    tap_indices_Shw = np.arange(len(Shw))
    markerline1, stemlines1, baseline1 = axes1[1].stem(tap_indices_Sw, Sw, label='Coefficients of S(z)', basefmt=' ')
    markerline2, stemlines2, baseline2 = axes1[1].stem(tap_indices_Shw, Shw, label='Coefficients of Sh(z)', 
                                                        linefmt='r-', markerfmt='r*', basefmt=' ')
    axes1[1].set_ylabel('Amplitude')
    axes1[1].set_xlabel('Filter tap index')
    axes1[1].set_title('Secondary Path: Actual S(z) vs Estimated Sh(z)')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 2: Active Noise Control Results
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Noise residue over time
    axes2[0].plot(time_axis, e_cont, label='Noise residue e(k)', alpha=0.8)
    axes2[0].set_ylabel('Amplitude')
    axes2[0].set_xlabel('Discrete time k')
    axes2[0].set_title('Active Noise Control - Residual Noise')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Subplot 2: Noise signal vs Control signal
    axes2[1].plot(time_axis, y_primary, label='Noise signal yp(k)', alpha=0.7)
    axes2[1].plot(time_axis, y_control, 'r:', label='Control signal ys(k)', alpha=0.9)
    axes2[1].set_ylabel('Amplitude')
    axes2[1].set_xlabel('Discrete time k')
    axes2[1].set_title('Primary Noise vs Anti-Noise Control Signal')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 3: Additional analysis - Convergence and performance
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Learning curves (MSE over time using moving average)
    window_size = 50
    mse_iden = np.convolve(e_iden**2, np.ones(window_size)/window_size, mode='valid')
    mse_cont = np.convolve(e_cont**2, np.ones(window_size)/window_size, mode='valid')
    
    axes3[0].semilogy(np.arange(len(mse_iden)), mse_iden, label='Identification MSE', alpha=0.8)
    axes3[0].semilogy(np.arange(len(mse_cont)), mse_cont, label='Control MSE', alpha=0.8)
    axes3[0].set_ylabel('Mean Squared Error')
    axes3[0].set_xlabel('Discrete time k')
    axes3[0].set_title('Learning Curves (Moving Average MSE)')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    # Subplot 2: Controller weight evolution (show final weights)
    tap_indices_Cw = np.arange(len(Cw))
    markerline3, stemlines3, baseline3 = axes3[1].stem(tap_indices_Cw, Cw, label='Controller C(z) weights', basefmt=' ')
    axes3[1].set_ylabel('Amplitude')
    axes3[1].set_xlabel('Filter tap index')
    axes3[1].set_title('Final Controller Weights C(z)')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
