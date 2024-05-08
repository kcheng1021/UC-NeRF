import gin
import subprocess
import os
from torch.utils.tensorboard import SummaryWriter


@gin.configurable()
class Logger:
    def __init__(self, model, scheduler, output_file=None, SUM_FREQ=100):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}

        self.writer = SummaryWriter(log_dir=f'runs/{model.name}')
        # write some unbuffered output file, (for users with network problem with tunneling tensorboard)
        self.output = output_file
        self.SUM_FREQ = SUM_FREQ
        if not self.output is None:
            if not os.path.isdir('runs'):
                os.mkdir('runs')
            f1 = open(os.path.join("runs", self.output), "w")
            f1.close()

    def _print_training_status(self):
        SUM_FREQ = self.SUM_FREQ
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")
        if not self.output is None:
            f = open(os.path.join("runs", self.output), "a")
            f.write(f"{training_str + metrics_str}\n")
            f.close()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        SUM_FREQ = self.SUM_FREQ
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}


    def close(self):
        self.writer.close()

        