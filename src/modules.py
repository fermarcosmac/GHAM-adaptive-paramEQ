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
        # TODO: this has to be a Torch Parameter!
        self.current_params: torch.nn.Parameter = init_params_tensor                  # (1 x n_params) torch tensor
        self.state: dict = {
            "optimizer": torch.optim.SGD([self.current_params], lr=0.01)
        }                                                   # dictionary to hold any state information
        self.prev_params: torch.Tensor = torch.zeros_like(init_params_tensor)   # (1 x n_params) torch tensor
        self.method: str = "TD-FxLMS"                                           # adaptation method (placeholder)

    # Dont use getters or setters! Just make sure properties are public.
    
    def update(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor):
        if self.method == "TD-FxLMS":
            self._update_TD_FxLMS(in_frame, EQed_frame, out_frame)
        else:   # placeholder (no adaptation logic)
            new_params = self.current_params
            self.prev_params = self.current_params
            self.current_params = new_params

    def _update_TD_FxLMS(self, in_frame: torch.Tensor, EQed_frame: torch.Tensor, out_frame: torch.Tensor):
        # Placeholder adaptation logic for TD-FxLMS
        learning_rate = 0.01
        loss = torch.functional.mse_loss(EQed_frame, out_frame)
        loss.backward()

        new_params = self.current_params - learning_rate * gradient
        self.prev_params = self.current_params
        self.current_params = new_params


# TODO
class EQLogger:
    def __init__(self):
        pass