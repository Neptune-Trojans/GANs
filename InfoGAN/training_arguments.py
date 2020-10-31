from dataclasses import dataclass


@dataclass(frozen=True)
class Arguments:
    batch_size: int = 256
    buffer_size: int = 60000
    noise_dim: int = 89
    num_classes: int = 10
    num_examples_to_generate: int = 16
    epochs: int = 256
    init_lr: float = 0.0002
    visualization_folder: str = './training_visualization'

    @property
    def epoch_iteration_steps(self):
        return self.buffer_size // self.batch_size

    @property
    def total_iteration_steps(self):
        return self.buffer_size // self.batch_size * self.epochs
