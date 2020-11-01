from dataclasses import dataclass


@dataclass(frozen=True)
class Arguments:
    batch_size: int = 256
    buffer_size: int = 60000
    noise_dim: int = 89
    num_classes: int = 10
    num_examples_to_generate: int = 25
    epochs: int = 256
    init_lr: float = 0.0002
    visualization_folder: str = './training_visualization'
    check_points: str = './check_points'

    @property
    def epoch_iteration_steps(self):
        # while training discriminator we use half real images and half synthetic noise.
        return self.buffer_size // self.batch_size * 2

    @property
    def total_iteration_steps(self):
        return self.epoch_iteration_steps * self.epochs
