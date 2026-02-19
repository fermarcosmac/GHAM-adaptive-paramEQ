import torch
import torchaudio
from torch.func import jacrev


class LEMConv(torch.autograd.Function):
    """Minimal copy of LEMConv from experiment_03.

    - forward: uses true LEM impulse response (h_true)
    - backward: uses estimated LEM impulse response (h_est) via FFT-based convolution
    """

    @staticmethod
    def forward(x, h_true, h_est):
        """Forward pass using true LEM impulse response.

        Args:
            x:      (B, 1, N) input signal segment
            h_true: (1, 1, M) true LEM IR (no grad)
            h_est:  (1, 1, M) estimated LEM IR used only for gradients
        """
        # NOTE: ctx is *not* touched here; functorch will call setup_context.
        y = torchaudio.functional.fftconvolve(x, h_true, mode="full")
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save context for backward in a functorch-compatible way."""
        x, h_true, h_est = inputs
        ctx.save_for_backward(h_est)
        ctx.input_len = x.shape[-1]
        ctx.h_est_len = h_est.shape[-1]

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using estimated LEM impulse response via FFT-based convolution."""
        (h_est,) = ctx.saved_tensors
        N = ctx.input_len
        M = ctx.h_est_len

        grad_full = torchaudio.functional.fftconvolve(
            grad_output, torch.flip(h_est, dims=[-1]), mode="full"
        )
        grad_x = grad_full[..., M - 1 : M - 1 + N]

        return grad_x, None, None


def toy_loss(EQ_params, x, h_true, h_est):
    """Toy loss mimicking the mechanics of experiment_03 around LEMConv.

    - Simple "EQ": scale the input by EQ_params (broadcast over time).
    - Pass through LEMConv with true IR in forward, estimated IR in backward.
    - Compute MSE vs a fixed target.
    """
    # EQ: just a scalar gain on x for simplicity
    gain = EQ_params.view(1, 1, 1)
    x_eq = gain * x

    # LEMConv: forward uses h_true, backward uses h_est
    y = LEMConv.apply(x_eq, h_true, h_est)

    # Target: here just zeros to keep things simple
    target = torch.zeros_like(y)

    loss = torch.mean((y - target) ** 2)
    return loss


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple 1D input and IRs
    N = 128
    M = 33

    x = torch.randn(1, 1, N, device=device)
    h_true = torch.randn(1, 1, M, device=device)

    # Estimated IR for backward (different from true IR to mimic experiment_03)
    h_est = torch.randn(1, 1, M, device=device)

    # Single scalar EQ parameter
    EQ_params = torch.tensor([1.0], device=device, requires_grad=True)

    # Standard backward as a cross-check
    print("\nChecking standard backward()...")
    loss = toy_loss(EQ_params, x, h_true, h_est)
    loss.backward()
    print("grad via backward:", EQ_params.grad)

    # Define jacobian function w.r.t. EQ_params only
    jac_fcn = jacrev(toy_loss, argnums=0)

    print("Running jacrev on toy_loss...")
    jac = jac_fcn(EQ_params, x, h_true, h_est)

    print("Jacobian shape:", jac.shape)
    print("Jacobian value:", jac)


if __name__ == "__main__":
    main()
