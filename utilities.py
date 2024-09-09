import torch
import time


def assign_fixed_params(model):
    rng_gen = torch.Generator()
    rng_gen.manual_seed(123)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(*p.shape, generator=rng_gen, dtype=p.dtype))

def save_to_files(models, losses, accuracies):
    for i,model in enumerate(models):
        torch.save(model.state_dict(), f"atomic_models/model_{i}.pth")

    np.save("atomic_models/losses.npy", losses)
    np.save("atomic_models/accuracies.npy", accuracies)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

    def pstop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:.4f} seconds")