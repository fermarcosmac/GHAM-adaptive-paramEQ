import torch

# Import LEMConv from your main experiment script
from experiment_03 import LEMConv


def run_debug(device: str = "cpu") -> None:
    """Minimal test for stepping into LEMConv.backward.

    Run this file under the VS Code debugger, set a breakpoint inside
    LEMConv.backward (in experiment_03.py), and then call run_debug().
    """
    dev = torch.device(device)

    # Small toy shapes
    B, C = 1, 1
    N = 256   # input length
    M = 128   # IR length

    # Input signal (requires grad so backward has something to compute)
    x = torch.randn(B, C, N, device=dev, dtype=torch.float32, requires_grad=True)

    # True and estimated LEM IRs
    h_true = torch.randn(1, 1, 2*M, device=dev, dtype=torch.float32)
    h_est = torch.randn(1, 1, M, device=dev, dtype=torch.float32)

    # Forward through custom autograd Function
    y = LEMConv.apply(x, h_true, h_est)

    # Simple scalar loss so autograd runs backward
    loss = (y ** 2).sum()
    print(f"Loss value: {loss.item():.6f}")

    # Place a breakpoint inside LEMConv.backward (experiment_03.py)
    # and run this script under the debugger; when loss.backward() is
    # called, the breakpoint should be hit.
    loss.backward()

    print("Gradient wrt x (x.grad) stats:")
    print(f"  mean: {x.grad.mean().item():.6e}")
    print(f"  std:  {x.grad.std().item():.6e}")


if __name__ == "__main__":
    # Change to "cuda" if you want to test on GPU and it is available.
    run_debug(device="cpu")
