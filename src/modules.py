import sys
from pathlib import Path
import torch
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))

#TODO
class EQController_dasp:
    def __init__(self, EQ, init_params_tensor):
        self.EQ = EQ                                                            # ParametricEQ object (dasp_pytorch)
        self.current_params: torch.Tensor = init_params_tensor                  # (1 x n_params) torch tensor
        self.prev_params: torch.Tensor = torch.zeros_like(init_params_tensor)   # (1 x n_params) torch tensor

    # Dont use getters or setters! Just make sure properties are public.
    
    def update_params(self):
        new_params = self.current_params # placeholder (so far no adaptation logic)
        self.prev_params = self.current_params
        self.current_params = new_params


# TODO
class EQLogger:
    def __init__(self):
        pass