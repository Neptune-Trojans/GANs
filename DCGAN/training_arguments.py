from enum import Enum

from dataclasses import dataclass


class SupportedModels(Enum):
    CIFAR10 = 'CIFAR10'
    MNIST = 'MNIST'


@dataclass(frozen=True)
class Arguments:
    batch_size: int = 256
    buffer_size: int = 60000
    noise_dim: int = 100
    num_examples_to_generate: int = 16
    epochs: int = 256
    init_lr: float = 1e-4
    visualization_folder: str = './training_visualization'
    model: str = SupportedModels.CIFAR10

    @property
    def epoch_iteration_steps(self):
        return self.buffer_size // self.batch_size

    @property
    def total_iteration_steps(self):
        return self.buffer_size // self.batch_size * self.epochs
