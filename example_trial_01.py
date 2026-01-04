#import os
# Temporary workaround: allow duplicate OpenMP runtime while debugging.
# Unsafe: may lead to crashes or incorrect results. See notes below.
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#I have a GeForce RTX 4060 GPU, and the NVIDIA webpage ays that the Compute Capability (CC) for this GPU is 8.9. So please, install cuda and 

import torch
import torchaudio
import dasp_pytorch
import matplotlib.pyplot as plt

# Load audio
x, sr = torchaudio.load("audio/guitar-riff.wav")

# create batch dim
# (batch_size, n_channels, n_samples)
x = x.unsqueeze(0)

# apply some distortion with 16 dB drive
drive = torch.tensor([16.0])
y = dasp_pytorch.functional.distortion(x, sr, drive)

# create a parameter to optimizer
drive_hat = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([drive_hat], lr=0.01)

# optimize the parameter
n_iters = 2500
for n in range(n_iters):
    # apply distortion with the estimated parameter
    y_hat = dasp_pytorch.functional.distortion(x, sr, drive_hat)

    # compute distance between estimate and target
    loss = torch.nn.functional.mse_loss(y_hat, y)

    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"step: {n+1}/{n_iters}, loss: {loss.item():.3e}, drive: {drive_hat.item():.3f}\r"
    )

# Plot original, distorted, and processed signals on the same axis for comparison
plt.figure(figsize=(12, 4))
plt.plot(x.squeeze().cpu().numpy(), label="Original")
plt.plot(y.squeeze().cpu().numpy(), label="Distorted")
plt.plot(y_hat.squeeze().detach().cpu().numpy(), label="Processed")
plt.title("Signals Comparison")
plt.legend()
plt.tight_layout()
plt.show()