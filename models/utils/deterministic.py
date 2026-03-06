"""
Deterministic inference utilities.
Ensures identical outputs for identical inputs.
"""
import torch
import numpy as np
import random
import os


GLOBAL_SEED = 42


def set_deterministic(seed: int = GLOBAL_SEED):
    """
    Set all random seeds to ensure deterministic behavior.
    Must be called before any inference.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def inference_context(model):
    """
    Context manager that ensures deterministic inference.
    Sets model to eval mode and disables gradients.
    """
    model.eval()
    return torch.no_grad()


class DeterministicGuard:
    """
    Context manager to enforce determinism around any block of code.
    """
    def __init__(self, seed: int = GLOBAL_SEED):
        self.seed = seed

    def __enter__(self):
        set_deterministic(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def validate_determinism(inference_fn, input_data, runs: int = 3) -> bool:
    """
    Validate that an inference function produces identical results
    across multiple runs with the same input.
    
    Returns True if all runs produce identical results.
    """
    results = []
    for _ in range(runs):
        set_deterministic(GLOBAL_SEED)
        result = inference_fn(input_data)
        results.append(result)

    # Check all results are identical
    for i in range(1, len(results)):
        if abs(results[0]['confidence'] - results[i]['confidence']) > 1e-6:
            return False
        if results[0]['prediction'] != results[i]['prediction']:
            return False
    return True
