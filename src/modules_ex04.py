import torch, torchaudio


class LEMConv(torch.autograd.Function):
    """Custom LEM convolution with functorch-compatible context and jvp.

    - forward: uses true LEM impulse response (h_true)
    - backward: uses estimated LEM impulse response (h_est) via FFT-based convolution
    """

    # Let PyTorch generate a vmap rule for forward-mode AD
    generate_vmap_rule = True

    @staticmethod
    def forward(x, h_true, h_est):
        """Forward pass using true LEM impulse response.

        Args:
            x:      (B, 1, N) input signal segment
            h_true: (1, 1, M) true LEM IR (no grad)
            h_est:  (1, 1, M) estimated LEM IR used only for gradients
        """
        # Do not touch ctx here; functorch will call setup_context separately.
        y = torchaudio.functional.fftconvolve(x, h_true, mode="full")
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save context for backward/jvp in a functorch-compatible way."""
        x, h_true, h_est = inputs
        ctx.save_for_backward(h_est)
        ctx.input_len = x.shape[-1]
        ctx.h_est_len = h_est.shape[-1]
        # For jvp
        ctx.h_true = h_true
        ctx.x_shape = x.shape

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

    @staticmethod
    def jvp(ctx, x_t, h_true_t, h_est_t):
        """Jacobian-vector product for forward-mode AD.

        Signature: jvp(ctx, *grad_inputs) -> *grad_outputs.
        We treat only x as differentiable; h_true and h_est are constants.
        """
        h_true = ctx.h_true

        if x_t is None:
            x_t = torch.zeros(ctx.x_shape, device=h_true.device, dtype=h_true.dtype)

        y_t = torchaudio.functional.fftconvolve(x_t, h_true, mode="full")

        # Single output -> single tangent
        return y_t