import yaml, torch

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()