import os
import sys
import torch
import torchaudio
# Ensure the workspace root is first on sys.path so the local package is imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import my_dasp_pytorch as dasp_pytorch
import matplotlib.pyplot as plt

# Load audio
x, sr = torchaudio.load("audio/guitar-riff.wav")

# create batch dim
# (batch_size, n_channels, n_samples)
x = x.unsqueeze(0)

# Define known EQ parameters
low_shelf_gain_db       = torch.tensor([-10.0])
low_shelf_cutoff_freq   = torch.tensor([100.0])
low_shelf_q_factor      = torch.tensor([1.0])
band0_gain_db           = torch.tensor([10.0])
band0_cutoff_freq       = torch.tensor([500.0])
band0_q_factor          = torch.tensor([1.0])
band1_gain_db           = torch.tensor([-10.0])
band1_cutoff_freq       = torch.tensor([1000.0])
band1_q_factor          = torch.tensor([1.0])
band2_gain_db           = torch.tensor([10.0])
band2_cutoff_freq       = torch.tensor([2000.0])
band2_q_factor          = torch.tensor([1.0])
band3_gain_db           = torch.tensor([-10.0])
band3_cutoff_freq       = torch.tensor([4000.0])
band3_q_factor          = torch.tensor([1.0])
high_shelf_gain_db      = torch.tensor([10.0])
high_shelf_cutoff_freq  = torch.tensor([8000.0])
high_shelf_q_factor     = torch.tensor([1.0])

# Generate output data by EQ-ing the input with known parameters
y = dasp_pytorch.functional.parametric_eq(x, sr,
                                        low_shelf_gain_db,
                                        low_shelf_cutoff_freq,
                                        low_shelf_q_factor,
                                        band0_gain_db,
                                        band0_cutoff_freq,
                                        band0_q_factor,
                                        band1_gain_db,
                                        band1_cutoff_freq,
                                        band1_q_factor,
                                        band2_gain_db,
                                        band2_cutoff_freq,
                                        band2_q_factor,
                                        band3_gain_db,
                                        band3_cutoff_freq,
                                        band3_q_factor,
                                        high_shelf_gain_db,
                                        high_shelf_cutoff_freq,
                                        high_shelf_q_factor)

hey = 0