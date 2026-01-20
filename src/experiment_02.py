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
from utils import (
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
)


#%% MAIN SCRIPT

if __name__ == "__main__":

    #%% CONFIGURATION AND SETUP
    
    # Set paths (same as experiment_01)
    base = Path(".")
    rir_dir = base / "data" / "rir"
    
    # Set simulation parameters
    sr = 48000  # Sample rate (will be updated if RIR has different rate)
    T_seconds = 4.0  # Simulation duration in seconds
    
    # Load RIR for secondary path S(z)
    # The RIR represents the acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=1)
    
    # Use the first RIR's sample rate as reference
    sr = rirs_srs[0]
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)
    rir = rirs[0]  # Use first RIR
    
    # S(z): Secondary path coefficients from loaded RIR
    # Truncate or use full RIR based on desired filter length
    Sw_max_len = min(len(rir), 8192//2)  # Limit secondary path length for computational efficiency
    Sw = rir[:Sw_max_len].astype(np.float64)  # Secondary path coefficients
    Sw = Sw / np.max(np.abs(Sw))  # Normalize to prevent numerical issues
    
    # Estimate delay from the RIR (position of first peak = direct sound arrival)
    est_delay = get_delay_from_ir(rir, sr)
    est_delay += 128 # Hardcoded! Half the compensating filter length
    print(f"Estimated delay from RIR: {est_delay} samples ({est_delay/sr*1000:.2f} ms)")
    
    # P(z): Primary path is modeled as a pure delay
    # This represents direct sound propagation without room reflections
    primary_delay = est_delay
    
    # Set simulation duration (number of samples)
    T = int(T_seconds * sr)
    
    print("=" * 60)
    print("FxLMS Algorithm with Real RIR")
    print("=" * 60)
    print(f"Sample rate: {sr} Hz")
    print(f"Simulation duration: {T_seconds} s ({T} samples)")
    print(f"RIR length: {len(rir)} samples ({len(rir)/sr*1000:.2f} ms)")
    print(f"Secondary path S(z) length: {len(Sw)} samples ({len(Sw)/sr*1000:.2f} ms)")
    print(f"Primary path P(z): Pure delay of {primary_delay} samples ({primary_delay/sr*1000:.2f} ms)")

    #%% PART 1: SECONDARY PATH IDENTIFICATION
    # The first task is to estimate S(z) using the LMS algorithm
    
    print("\n" + "=" * 60)
    print("PART 1: Secondary Path Identification")
    print("=" * 60)
    
    # Generate white noise for identification
    np.random.seed(42)  # For reproducibility
    x_iden = np.random.randn(T)
    
    # Send the noise through the actual secondary path S(z)
    y_iden = np.convolve(x_iden, Sw, mode='full')[:T]  # Truncate to T samples
    
    # Initialize the secondary path estimate Sh(z)
    # Use a filter length that can capture the main features of the RIR
    Sh_len = min(1024, len(Sw))  # Filter length for Sh(z) - balance between accuracy and computation
    Shx = np.zeros(Sh_len)  # State buffer (input samples)
    Shw = np.zeros(Sh_len)  # Filter weights/coefficients
    e_iden = np.zeros(T)    # Identification error buffer
    
    # Apply vanilla LMS algorithm for identification
    # Learning rate needs to be smaller for longer filters and real RIRs
    mu_iden = 0.01 / Sh_len  # Normalized learning rate
    
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
    # Primary path is modeled as a pure delay (direct sound propagation)
    y_primary = np.zeros(T)
    if primary_delay < T:
        y_primary[primary_delay:] = input[:T - primary_delay]
    
    # Initialize the controller C(z)
    # Controller length should be long enough to model the inverse of secondary path
    C_len = min(256, Sh_len)  # Controller filter length
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
    # Learning rate needs to be smaller for real RIRs
    mu_cont = 0.001 / C_len  # Normalized learning rate for controller
    
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
    
    # Create time axis in seconds for better visualization
    time_axis = np.arange(T) / sr
    time_axis_samples = np.arange(T)
    
    # Plot 1: Secondary Path Identification Results
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Identification error over time
    axes1[0].plot(time_axis, e_iden, label='Identification error', alpha=0.8)
    axes1[0].set_ylabel('Amplitude')
    axes1[0].set_xlabel('Time (s)')
    axes1[0].set_title('Secondary Path Identification Error')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Subplot 2: Comparison of S(z) and Sh(z) coefficients (first Sh_len samples)
    # Only compare the portion that Sh(z) can model
    tap_time_Sw = np.arange(min(len(Sw), Sh_len)) / sr * 1000  # Time in ms
    tap_time_Shw = np.arange(len(Shw)) / sr * 1000  # Time in ms
    axes1[1].plot(tap_time_Sw, Sw[:len(tap_time_Sw)], 'b-', label='S(z) - Actual RIR', alpha=0.7)
    axes1[1].plot(tap_time_Shw, Shw, 'r--', label='Sh(z) - Estimated', alpha=0.9)
    axes1[1].axvline(x=est_delay/sr*1000, color='g', linestyle=':', label=f'Estimated delay ({est_delay/sr*1000:.2f} ms)')
    axes1[1].set_ylabel('Amplitude')
    axes1[1].set_xlabel('Time (ms)')
    axes1[1].set_title(f'Secondary Path: Actual RIR vs Estimated Sh(z) (first {Sh_len} taps)')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 2: Active Noise Control Results
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Noise residue over time
    axes2[0].plot(time_axis, e_cont, label='Noise residue e(k)', alpha=0.8)
    axes2[0].set_ylabel('Amplitude')
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_title('Active Noise Control - Residual Noise')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Subplot 2: Noise signal vs Control signal
    axes2[1].plot(time_axis, y_primary, label='Noise signal yp(k) (delayed input)', alpha=0.7)
    axes2[1].plot(time_axis, y_control, 'r:', label='Control signal ys(k)', alpha=0.9)
    axes2[1].set_ylabel('Amplitude')
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_title(f'Primary Noise (delay={primary_delay} samples) vs Anti-Noise Control Signal')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Plot 3: Additional analysis - Convergence and performance
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Learning curves (MSE over time using moving average)
    window_size = int(0.01 * sr)  # 10ms moving average window
    if window_size < 10:
        window_size = 10
    mse_iden = np.convolve(e_iden**2, np.ones(window_size)/window_size, mode='valid')
    mse_cont = np.convolve(e_cont**2, np.ones(window_size)/window_size, mode='valid')
    mse_time_axis = np.arange(len(mse_iden)) / sr
    
    axes3[0].semilogy(mse_time_axis, mse_iden, label='Identification MSE', alpha=0.8)
    axes3[0].semilogy(np.arange(len(mse_cont)) / sr, mse_cont, label='Control MSE', alpha=0.8)
    axes3[0].set_ylabel('Mean Squared Error')
    axes3[0].set_xlabel('Time (s)')
    axes3[0].set_title(f'Learning Curves (Moving Average MSE, window={window_size/sr*1000:.1f}ms)')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    # Subplot 2: Controller weight evolution (show final weights)
    tap_time_Cw = np.arange(len(Cw)) / sr * 1000  # Time in ms
    axes3[1].plot(tap_time_Cw, Cw, 'b-', label='Controller C(z) weights')
    axes3[1].set_ylabel('Amplitude')
    axes3[1].set_xlabel('Time (ms)')
    axes3[1].set_title(f'Final Controller Weights C(z) ({C_len} taps)')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
