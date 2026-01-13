import sys
from pathlib import Path
import torch
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
import matplotlib.pyplot as plt

#TODO
class EQController_dasp:
    def __init__(self, EQ, init_params_tensor):
        self.EQ = EQ                                                            # ParametricEQ object (dasp_pytorch)
        # Create as nn.Parameter so it's optimizable by PyTorch optimizer
        self.params = torch.nn.Parameter(init_params_tensor.clone().detach().requires_grad_(True))  # (1 x n_params) torch Parameter
        self.state: dict = {}                                                   # dictionary to hold any state information
        self.prev_params: torch.Tensor = torch.zeros_like(self.params)   # (1 x n_params) torch tensor
        self.method: str = "TD-FxLMS"                                           # adaptation method (placeholder)
        # The optimizer should depend on the chosen method TODO
        self.optimizer = torch.optim.SGD([self.params], lr=0.01)        # optimizer for parameter updates

    # Dont use getters or setters! Just make sure properties are public.
    
    def update(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor):
        if self.method == "TD-FxLMS":
            self._update_TD_FxLMS(in_frame, EQed_frame, out_frame)
        else:   # placeholder (no adaptation logic)
            new_params = self.params
            self.prev_params = self.params
            self.params = new_params

    def _update_TD_FxLMS(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor):
        # Placeholder adaptation logic for TD-FxLMS

        # Track parameters prior to step
        self.prev_params = self.params.clone()

        # Match length of EQed_frame to out_frame if needed (only for loss evaluation)
        min_len = min(in_frame.shape[-1], out_frame.shape[-1])

        # TODO: WE NEED A DELAYED VERSION OF THE INPUT TO DO TIME DOMAIN LOSS

        # Compute Time-Domain loss
        loss = torch.nn.functional.mse_loss(in_frame[..., :min_len], out_frame[..., :min_len])

        # Step the pytorch optimizer (updates self.params)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clip parameters to [0, 1] range to ensure normalized values (preserves gradient flow)
        with torch.no_grad():
            self.params.clamp_(0.0, 1.0)


# TODO
class EQLogger:
    def __init__(self):
        pass