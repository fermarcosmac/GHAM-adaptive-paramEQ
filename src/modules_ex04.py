import torch, torchaudio

class LEMConv(torch.autograd.Function):
    """
    Custom LEM convolution:
    - forward: uses true LEM impulse response (h_true)
    - backward: uses estimated LEM impulse response (h_est) via FFT-based convolution
    """

    @staticmethod
    def forward(ctx, x, h_true, h_est):
        """Forward pass using true LEM impulse response.

        Args:
            x:      (B, 1, N) input signal segment
            h_true: (1, 1, M) true LEM IR (no grad)
            h_est:  (1, 1, M) estimated LEM IR used only for gradients
        """
        y = torchaudio.functional.fftconvolve(x, h_true, mode="full")
        ctx.save_for_backward(h_est)
        ctx.input_len = x.shape[-1]
        #ctx.LEM_len = h_true.shape[-1]
        ctx.h_est_len = h_est.shape[-1]
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using estimated LEM impulse response via FFT-based convolution."""
        (h_est,) = ctx.saved_tensors
        N = ctx.input_len
        M = ctx.h_est_len

        grad_full = torchaudio.functional.fftconvolve(grad_output, torch.flip(h_est, dims=[-1]), mode="full")
        grad_x = grad_full[..., M-1:M-1+N]

        return grad_x, None, None