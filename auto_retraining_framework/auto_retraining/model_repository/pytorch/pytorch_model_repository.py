import os
import torch

from ..model_repository import ModelRepository
from typing import Any

current_path = os.getcwd()
ROOT_DIR = os.path.join(current_path, ".repository")

class PytorchModelRepository(ModelRepository):
    def __init__(self, root_dir = ROOT_DIR, experiment_name: str = None):
        self.root_dir = root_dir
        self.experiment_name = experiment_name
    
    
    def write(self, model: Any, model_name: str, model_tag: str) -> None:
        model_file_name = f"{model_tag}.pt"
        model_path = os.path.join(self.root_dir, self.experiment_name, model_name)
        model_file_path = os.path.join(model_path, model_file_name)
        if os.path.exists(model_file_path):
            raise FileExistsError(f"Model {model_name}/{model_tag} already exists at {model_path}")
        os.makedirs(model_path, exist_ok=True)
        torch.save(model, model_file_path)
        latest_path = os.path.join(model_path, "latest.pt")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_file_path, latest_path)

    
    def load(self, model_name: str, model_tag: str, device: str = "cpu") -> Any:
        model_name_with_tag = f"{model_name}_{model_tag}"
        model_path = os.path.join(self.root_dir, self.experiment_name, model_name_with_tag)
        model_path = os.path.join(model_path, "model.pt")
        return torch.load(model_path, map_location=torch.device(device))


    def load_latest(self, model_name: str, device: str = "cpu") -> Any:
        model_path = os.path.join(self.root_dir, self.experiment_name, model_name)
        latest_path = os.path.join(model_path, "latest.pt")
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"Latest model for {model_name} not found at {latest_path}")
        return torch.load(latest_path, map_location=torch.device(device))
