# Initial room impulse response measurements

# Parametric EQ estimation for virtual room compensation

# Playback simulation:
#    - Static room - no EQ
#    - Static room + initial EQ
#    - Dynamic room - no EQ
#    - Dynamic room + adaptive EQ

# Common interface for different adaptive EQ settings (my class, torch.optim?)
#    - Pytorch optimizers (including Newton library)
#    - GHAM adaptive filter
#    - Neural network-based adaptive EQ (MLP/RNN)

# They should take in the error signal and output EQ parameters at each correction step
# Make sure each correctioin step fits in real-time constraints (e.g., 10 ms processing time budget)