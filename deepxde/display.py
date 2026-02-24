import sys
from datetime import datetime

import numpy as np

from .utils import list_to_str


class TrainingDisplay:
    """Display training progress."""

    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        # YYYY-MM-DD HH:MM:SS -> 19 characters
        self.len_time = 22
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4, s5):
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}{:{l5}s}".format(
                s1,
                s2,
                s3,
                s4,
                s5,
                l1=self.len_time,
                l2=10,
                l3=self.len_train,
                l4=self.len_test,
                l5=self.len_metric,
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Time", "Step", "Train loss", "Test loss", "Test metric")
        self.is_header_print = True

    def __call__(self, train_state):
        if not self.is_header_print:
            self.len_train = len(train_state.loss_train) * 10 + 4
            self.len_test = len(train_state.loss_test) * 10 + 4
            self.len_metric = len(train_state.metrics_test) * 10 + 4
            self.header()
        self.print_one(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(train_state.step),
            list_to_str(train_state.loss_train),
            list_to_str(train_state.loss_test),
            list_to_str(train_state.metrics_test),
        )

    def summary(self, train_state):
        print("Best model at step {:d}:".format(train_state.best_step))
        print("  train loss: {:.2e}".format(train_state.best_loss_train))
        print("  test loss: {:.2e}".format(train_state.best_loss_test))
        print("  test metric: {:s}".format(list_to_str(train_state.best_metrics)))
        if train_state.best_ystd is not None:
            print("  Uncertainty:")
            print("    l2: {:g}".format(np.linalg.norm(train_state.best_ystd)))
            print(
                "    l_infinity: {:g}".format(
                    np.linalg.norm(train_state.best_ystd, ord=np.inf)
                )
            )
            print(
                "    max uncertainty location:",
                train_state.X_test[np.argmax(train_state.best_ystd)],
            )
        print("")
        self.is_header_print = False


training_display = TrainingDisplay()
