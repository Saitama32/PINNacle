import logging
import os
import csv
import torch
import numpy as np
import scipy
import itertools
import copy
from deepxde.geometry import Hypercube, Interval
from deepxde.callbacks import Callback
from src.utils import plot
import scipy.interpolate
import deepxde as dde
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.utils.ks_metrics import long_horizon_metrics

logger = logging.getLogger(__name__)


def _float_to_str(value):
    return f"{float(value):.10e}"


def _array_to_str(values):
    return "[" + ", ".join(_float_to_str(value) for value in np.ravel(values)) + "]"


class PlotCallback(Callback):

    def __init__(self, log_every=None, verbose=False, fast=False):
        super(PlotCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fast = fast
        self.valid_epoch = 0

    def plot(self, save_path, save_true=False):
        train_state = self.model.train_state
        plot.plot_state(
            self.model.pde,
            train_state,
            save_path,
            is_best=False,
            fast=self.fast,
            save_true=save_true,
        )

    def on_train_begin(self):
        self.base_save_path = self.model.model_save_path + "/"
        if not os.path.exists(self.base_save_path):
            os.mkdir(self.base_save_path)

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.log_every is None or self.log_every <= 0:
            return
        if self.valid_epoch % self.log_every != 0:
            return
        if self.verbose:
            print("Plotting at epoch {} ...".format(self.valid_epoch))

        save_path = self.base_save_path + str(self.valid_epoch) + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.plot(save_path, save_true=(self.valid_epoch == self.log_every))

    def on_train_end(self):
        if self.verbose:
            print("Plotting at train end ...")
        self.plot(self.base_save_path)


class SolutionImageCallback(Callback):
    """Save prediction/error images for scalar 2D problems with reference data."""

    def __init__(
        self,
        output_dir,
        log_every=100,
        metrics_path=None,
        experiment=None,
        metric_step=None,
    ):
        super().__init__()
        self.output_dir = os.fspath(output_dir)
        self.log_every = int(log_every)
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        self.local_epoch = 0
        self.rmse = np.nan
        self.brmse = np.nan
        self.metrics_path = None if metrics_path is None else os.fspath(metrics_path)
        self.experiment = experiment
        self.metric_step = metric_step
        self.latest_metrics = None

    def on_train_begin(self):
        os.makedirs(self.output_dir, exist_ok=True)
        ref_data = getattr(self.model.pde, "ref_data", None)
        if ref_data is None:
            raise ValueError("SolutionImageCallback requires pde.ref_data")
        if self.model.pde.input_dim != 2 or self.model.pde.output_dim != 1:
            raise ValueError(
                "SolutionImageCallback supports only two-dimensional inputs "
                "with a scalar output"
            )
        ref_data = np.asarray(ref_data)
        self.test_x = ref_data[:, : self.model.pde.input_dim]
        self.test_y = ref_data[:, self.model.pde.input_dim :]

        if self.metrics_path is not None:
            metrics_dir = os.path.dirname(self.metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            self._prepare_metric_grid()
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "local_epoch",
                        "late_energy_agreement",
                        "late_spectral_overlap",
                        "late_normalized_wasserstein",
                    ]
                )

        bbox = np.asarray(self.model.pde.bbox, dtype=float)
        spatial_min, spatial_max = bbox[0], bbox[1]
        self.boundary_mask = np.isclose(self.test_x[:, 0], spatial_min) | np.isclose(
            self.test_x[:, 0], spatial_max
        )

    def _prepare_metric_grid(self):
        self.metric_x = np.unique(self.test_x[:, 0])
        self.metric_t = np.unique(self.test_x[:, 1])
        expected_size = len(self.metric_x) * len(self.metric_t)
        if len(self.test_x) != expected_size:
            raise ValueError(
                "Long-horizon metrics require a complete rectangular x/t reference grid"
            )
        self.metric_x_indices = np.searchsorted(self.metric_x, self.test_x[:, 0])
        self.metric_t_indices = np.searchsorted(self.metric_t, self.test_x[:, 1])
        occupancy = np.zeros((len(self.metric_t), len(self.metric_x)), dtype=np.int32)
        np.add.at(occupancy, (self.metric_t_indices, self.metric_x_indices), 1)
        if not np.all(occupancy == 1):
            raise ValueError("Reference x/t grid contains missing or duplicate points")
        self.exact_metric_field = np.empty(
            (len(self.metric_t), len(self.metric_x)), dtype=np.float64
        )
        self.exact_metric_field[self.metric_t_indices, self.metric_x_indices] = (
            self.test_y[:, 0]
        )

    def _record_metrics(self, prediction):
        if self.metrics_path is None:
            return
        prediction_field = np.empty_like(self.exact_metric_field)
        prediction_field[self.metric_t_indices, self.metric_x_indices] = prediction[:, 0]
        self.latest_metrics = long_horizon_metrics(
            prediction_field,
            self.exact_metric_field,
        )
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self.local_epoch,
                    self.latest_metrics["late_energy_agreement"],
                    self.latest_metrics["late_spectral_overlap"],
                    self.latest_metrics["late_normalized_wasserstein"],
                ]
            )
        print(
            "Long-horizon metrics: "
            f"epoch={self.local_epoch} "
            f"late_energy_agreement={self.latest_metrics['late_energy_agreement']:.10e} "
            f"late_spectral_overlap={self.latest_metrics['late_spectral_overlap']:.10e} "
            f"late_normalized_wasserstein="
            f"{self.latest_metrics['late_normalized_wasserstein']:.10e}"
        )

    def _save_images(self):
        prediction = np.asarray(self.model.predict(self.test_x))
        error = prediction - self.test_y
        self.rmse = float(np.sqrt(np.mean(error**2)))
        if np.any(self.boundary_mask):
            self.brmse = float(np.sqrt(np.mean(error[self.boundary_mask] ** 2)))
        self._record_metrics(prediction)

        prefix = f"epoch_{self.local_epoch:06d}"
        x = self.test_x[:, 0]
        t = self.test_x[:, 1]
        plot.plot_heatmap(
            x,
            t,
            prediction[:, 0],
            os.path.join(self.output_dir, f"{prefix}_prediction.png"),
            title=f"Prediction at local epoch {self.local_epoch}",
            xlabel="x",
            ylabel="t",
            pde=self.model.pde,
        )
        plot.plot_heatmap(
            x,
            t,
            error[:, 0],
            os.path.join(self.output_dir, f"{prefix}_error.png"),
            title=f"Error at local epoch {self.local_epoch}",
            xlabel="x",
            ylabel="t",
            pde=self.model.pde,
        )

    def on_epoch_end(self):
        self.local_epoch += 1
        if self.local_epoch % self.log_every == 0:
            self._save_images()

    def on_train_end(self):
        if self.local_epoch > 0 and self.local_epoch % self.log_every != 0:
            self._save_images()
        if self.experiment is not None and self.latest_metrics is not None:
            self.experiment.log_metrics(
                self.latest_metrics,
                step=self.metric_step,
            )


class LongHorizonMetricsCallback(Callback):
    """Log phase-insensitive metrics without creating solution images."""

    metric_names = (
        "late_energy_agreement",
        "late_spectral_overlap",
        "late_normalized_wasserstein",
    )

    def __init__(
        self,
        metrics_path,
        log_every=100,
        experiment=None,
        metric_step=None,
        epoch_offset=0,
    ):
        super().__init__()
        self.metrics_path = os.fspath(metrics_path)
        self.log_every = int(log_every)
        if self.log_every <= 0:
            raise ValueError("log_every must be a positive integer")
        self.experiment = experiment
        self.metric_step = metric_step
        self.epoch_offset = int(epoch_offset)
        self.local_epoch = 0
        self.latest_metrics = None
        self.rmse = np.nan
        self.brmse = np.nan

    def on_train_begin(self):
        ref_data = getattr(self.model.pde, "ref_data", None)
        if ref_data is None:
            raise ValueError("LongHorizonMetricsCallback requires pde.ref_data")
        ref_data = np.asarray(ref_data)
        input_dim = int(self.model.pde.input_dim)
        output_dim = int(self.model.pde.output_dim)
        if input_dim < 2:
            raise ValueError("Long-horizon metrics require space and time inputs")
        if ref_data.ndim != 2 or ref_data.shape[1] < input_dim + output_dim:
            raise ValueError("pde.ref_data has an incompatible shape")

        self.test_x = ref_data[:, :input_dim]
        self.test_y = ref_data[:, input_dim : input_dim + output_dim]
        bbox = np.asarray(self.model.pde.bbox, dtype=float)
        self.initial_mask = np.isclose(self.test_x[:, -1], bbox[-2])
        boundary_mask = np.zeros(len(self.test_x), dtype=bool)
        for dimension in range(input_dim - 1):
            boundary_mask |= np.isclose(
                self.test_x[:, dimension], bbox[2 * dimension]
            ) | np.isclose(self.test_x[:, dimension], bbox[2 * dimension + 1])
        self.boundary_mask = boundary_mask & ~self.initial_mask
        self.solution_l1 = float(np.mean(np.abs(self.test_y)))
        self.solution_l2 = float(np.sqrt(np.mean(self.test_y**2)))
        coordinate_values = [
            np.unique(self.test_x[:, dimension]) for dimension in range(input_dim)
        ]
        expected_size = int(np.prod([len(values) for values in coordinate_values]))
        if len(self.test_x) != expected_size:
            raise ValueError(
                "Long-horizon metrics require a complete rectangular reference grid"
            )

        coordinate_indices = [
            np.searchsorted(values, self.test_x[:, dimension])
            for dimension, values in enumerate(coordinate_values)
        ]
        # The PDE convention is [space..., time]; metric fields are [time, space...].
        self.field_indices = (
            coordinate_indices[-1],
            *coordinate_indices[:-1],
        )
        self.field_shape = (
            len(coordinate_values[-1]),
            *(len(values) for values in coordinate_values[:-1]),
        )
        occupancy = np.zeros(self.field_shape, dtype=np.uint8)
        np.add.at(occupancy, self.field_indices, 1)
        if not np.all(occupancy == 1):
            raise ValueError("Reference grid contains missing or duplicate points")

        self.exact_fields = np.empty(
            (*self.field_shape, output_dim), dtype=np.float64
        )
        self.exact_fields[self.field_indices] = self.test_y
        metrics_dir = os.path.dirname(self.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("local_epoch", *self.metric_names))

    def _record_metrics(self):
        prediction = np.asarray(self.model.predict(self.test_x), dtype=np.float64)
        if prediction.shape != self.test_y.shape:
            raise ValueError("Model prediction shape does not match reference outputs")
        error = prediction - self.test_y
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        mxe = float(np.max(np.abs(error)))
        self.rmse = float(np.sqrt(mse))
        if np.any(self.boundary_mask):
            boundary_mse = float(np.mean(error[self.boundary_mask] ** 2))
            self.brmse = float(np.sqrt(boundary_mse))
        else:
            boundary_mse = float("nan")
        if np.any(self.initial_mask):
            initial_mse = float(np.mean(error[self.initial_mask] ** 2))
        else:
            initial_mse = float("nan")
        l1_relative_error = mae / max(
            self.solution_l1, np.finfo(np.float64).eps
        )
        l2_relative_error = self.rmse / max(
            self.solution_l2, np.finfo(np.float64).eps
        )
        centered_rmse = float(np.abs(np.mean(error)))
        print(
            "Validation: epoch {} MSE {:.10e} MAE {:.10e} MXE {:.10e} "
            "BMSE {:.10e} ICMSE {:.10e} L1RE {:.10e} L2RE {:.10e} "
            "CRMSE {:.10e}".format(
                self.epoch_offset + self.local_epoch,
                mse,
                mae,
                mxe,
                boundary_mse,
                initial_mse,
                l1_relative_error,
                l2_relative_error,
                centered_rmse,
            )
        )
        prediction_fields = np.empty_like(self.exact_fields)
        prediction_fields[self.field_indices] = prediction
        component_metrics = [
            long_horizon_metrics(
                prediction_fields[..., component],
                self.exact_fields[..., component],
            )
            for component in range(self.exact_fields.shape[-1])
        ]
        self.latest_metrics = {
            name: float(np.mean([metrics[name] for metrics in component_metrics]))
            for name in self.metric_names
        }
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                (
                    self.local_epoch,
                    *(self.latest_metrics[name] for name in self.metric_names),
                )
            )
        summary = " ".join(
            f"{name}={self.latest_metrics[name]:.10e}" for name in self.metric_names
        )
        print(
            f"Long-horizon metrics: "
            f"epoch={self.epoch_offset + self.local_epoch} {summary}"
        )

    def on_epoch_end(self):
        self.local_epoch += 1
        if self.local_epoch % self.log_every == 0:
            self._record_metrics()

    def on_train_end(self):
        if self.local_epoch > 0 and self.local_epoch % self.log_every != 0:
            self._record_metrics()
        if self.experiment is not None and self.latest_metrics is not None:
            self.experiment.log_metrics(self.latest_metrics, step=self.metric_step)


class LossCallback(Callback):

    def __init__(self, verbose=False):
        super(LossCallback, self).__init__()
        self.log_every = None
        self.verbose = verbose
        self.valid_epoch = 0
        self.loss_weights = []

    def on_train_begin(self):
        self.log_every = self.model.display_every
        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights)
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))
            
    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.valid_epoch % self.log_every != 0:
            return

        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights.copy())
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))

        if self.verbose:
            loss_weight = self.loss_weights[-1]
            loss_train = self.model.losshistory.loss_train[-1] / loss_weight
            loss_test = self.model.losshistory.loss_test[-1] / loss_weight
            print('Unweighted Loss: {}  {} Weights: {}'.format(
                _array_to_str(loss_train),
                _array_to_str(loss_test),
                _array_to_str(loss_weight),
            ))

    def on_train_end(self):
        save_path = self.model.model_save_path + "/"
        loss_history = self.model.losshistory
        loss_weights = np.array(self.loss_weights)
        loss = np.hstack((
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train) / loss_weights,
            np.array(loss_history.loss_test) / loss_weights,
            loss_weights,
        ))
        np.savetxt(save_path + "loss.txt", loss, header="step, loss_train, loss_test, loss_weight")
        plot.plot_loss_history(self.model.pde, loss_history, save_path)
        plot.plot_loss_history(self.model.pde, loss_history, save_path, loss_weights=loss_weights)


class CausalDiagnosticsCallback(Callback):
    def __init__(self, log_every=None, verbose=False):
        super(CausalDiagnosticsCallback, self).__init__()
        self.log_every = log_every
        self.verbose = verbose
        self.rows = []
        self.keys = None
        self.epochs_since_last_log = 0

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every
        self.save_path = self.model.model_save_path + "/causal_diagnostics.txt"

    def on_epoch_end(self):
        self.epochs_since_last_log += 1
        if self.log_every is None or self.epochs_since_last_log < self.log_every:
            return
        self.epochs_since_last_log = 0

        diagnostics = getattr(self.model, "causal_loss_diagnostics", None)
        if not diagnostics:
            return

        if self.keys is None:
            self.keys = sorted(diagnostics.keys())

        row = [self.model.train_state.step]
        for key in self.keys:
            value = diagnostics.get(key, np.nan)
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            row.append(float(value))
        self.rows.append(row)

        if self.verbose:
            summary = ", ".join(
                f"{key}={row[i + 1]:.10e}" for i, key in enumerate(self.keys)
            )
            print(f"Causal diagnostics: step={row[0]}, {summary}")

    def on_train_end(self):
        if not self.rows:
            return
        header = "step, " + ", ".join(self.keys)
        np.savetxt(self.save_path, np.asarray(self.rows, dtype=float), header=header)


class IntegralDiagnosticsCallback(Callback):
    def __init__(self, log_every=None, verbose=False):
        super().__init__()
        self.log_every = log_every
        self.verbose = verbose
        self.rows = []
        self.epochs_since_last_log = 0
        self.keys = [
            "deepxde_loss_sum",
            "integral_loss_raw",
            "global_integral_loss",
            "global_integral_rms",
            "local_integral_loss",
            "local_integral_rms",
            "local_integral_mae",
            "local_integral_max",
            "local_normalized_rms",
            "local_normalized_mae",
            "local_normalized_max",
            "local_raw_mse",
            "local_normalized_mse",
            "local_mean_abs_raw_residual",
            "local_mean_abs_normalized_residual",
            "local_num_segments",
            "local_mean_segments_per_point",
            "local_max_segments_per_point",
            "local_mean_num_segments",
            "local_max_num_segments",
            "local_mean_segment_length",
            "local_min_segment_length",
            "local_max_segment_length",
            "local_chain_coverage_error",
            "local_chain_contiguity_error",
            "integral_weight",
            "weighted_global_integral_loss",
            "weighted_local_integral_loss",
            "integral_loss_weighted",
            "actual_total_loss",
            "integral_residual_rms",
            "integral_residual_abs_mean",
            "integral_residual_abs_max",
            "integral_loss_early",
            "integral_loss_middle",
            "integral_loss_late",
            "local_integral_loss_early",
            "local_integral_loss_middle",
            "local_integral_loss_late",
            "local_normalized_loss_early",
            "local_normalized_loss_middle",
            "local_normalized_loss_late",
        ]

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every
        self.save_path = self.model.model_save_path + "/integral_diagnostics.csv"

    def _value_to_float(self, value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def on_epoch_end(self):
        self.epochs_since_last_log += 1
        if self.log_every is None or self.epochs_since_last_log < self.log_every:
            return
        self.epochs_since_last_log = 0

        diagnostics = getattr(self.model, "integral_loss_diagnostics", None)
        if not diagnostics:
            return

        row = [self.model.train_state.step]
        for key in self.keys:
            row.append(self._value_to_float(diagnostics.get(key, np.nan)))
        self.rows.append(row)

        if self.verbose:
            values = dict(zip(self.keys, row[1:]))
            print(
                "[Integral loss] "
                f"step={row[0]} "
                f"deepxde={values['deepxde_loss_sum']:.10e} "
                f"raw={values['integral_loss_raw']:.10e} "
                f"global={values['global_integral_loss']:.10e} "
                f"local={values['local_integral_loss']:.10e} "
                f"weight={values['integral_weight']:.10e} "
                f"weighted={values['integral_loss_weighted']:.10e} "
                f"total={values['actual_total_loss']:.10e} "
                f"global_rms={values['global_integral_rms']:.10e} "
                f"local_rms={values['local_integral_rms']:.10e} "
                f"local_norm_rms={values['local_normalized_rms']:.10e} "
                f"early/middle/late="
                f"{values['integral_loss_early']:.10e}/"
                f"{values['integral_loss_middle']:.10e}/"
                f"{values['integral_loss_late']:.10e}"
            )

    def on_train_end(self):
        if not self.rows:
            return
        header = "step," + ",".join(self.keys)
        np.savetxt(
            self.save_path,
            np.asarray(self.rows, dtype=float),
            delimiter=",",
            header=header,
            comments="",
        )


class FrontIntegralDiagnosticsCallback(Callback):
    def __init__(self, log_every=None, verbose=False):
        super().__init__()
        self.log_every = log_every
        self.verbose = verbose
        self.rows = []
        self.epochs_since_last_log = 0
        self.keys = []

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every
        self.save_path = self.model.model_save_path + "/front_integral_diagnostics.csv"
        front_loss = self.model.front_integral_loss
        self.keys = [
            "deepxde_loss_sum",
            "front_integral_loss",
            "front_integral_weight",
            "weighted_front_integral_loss",
            "actual_total_loss",
            "front_defect_rms",
            "front_defect_max",
            "front_resample_count",
            "front_x_mean",
            "front_x_std",
        ]
        self.keys.extend(
            f"front_{index}_loss" for index in range(front_loss.num_intervals)
        )
        self.keys.extend(
            f"u_front_{index}_rms"
            for index in range(front_loss.num_intervals + 1)
        )

    @staticmethod
    def _value_to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def on_epoch_end(self):
        self.epochs_since_last_log += 1
        if self.log_every is None or self.epochs_since_last_log < self.log_every:
            return
        self.epochs_since_last_log = 0

        diagnostics = getattr(self.model, "front_integral_loss_diagnostics", None)
        if not diagnostics:
            return
        values = {
            key: self._value_to_float(diagnostics.get(key, np.nan))
            for key in self.keys
        }
        self.rows.append(
            [self.model.train_state.step] + [values[key] for key in self.keys]
        )
        if self.verbose:
            print(
                "[Front integral loss] "
                f"step={self.model.train_state.step} "
                f"raw={values['front_integral_loss']:.10e} "
                f"weight={values['front_integral_weight']:.10e} "
                f"weighted={values['weighted_front_integral_loss']:.10e} "
                f"rms={values['front_defect_rms']:.10e} "
                f"max={values['front_defect_max']:.10e}"
            )

    def on_train_end(self):
        if not self.rows:
            return
        header = "step," + ",".join(self.keys)
        np.savetxt(
            self.save_path,
            np.asarray(self.rows, dtype=float),
            delimiter=",",
            header=header,
            comments="",
        )


class KSDiagnosticsCallback(Callback):
    def __init__(self, log_every=None, chunk_every=1000, verbose=False):
        super().__init__()
        self.log_every = log_every
        self.chunk_every = chunk_every
        self.verbose = verbose
        self.rows = []
        self.keys = [
            "train_pde_mse",
            "validation_pde_mse",
            "grid_pde_mse",
            "ic_mse",
            "rms_u",
            "rms_u_t",
            "rms_alpha_u_u_x",
            "rms_beta_u_xx",
            "rms_gamma_u_xxxx",
            "rms_residual",
            "cancellation_ratio",
            "p99_abs_u_xxxx",
            "max_abs_u_xxxx",
            "max_u_xxxx_x",
            "max_u_xxxx_t",
            "periodic_gap_u",
            "periodic_gap_u_x",
            "periodic_gap_u_xx",
            "periodic_gap_u_xxx",
        ]
        self.chunk_keys = [
            "step",
            "chunk_id",
            "t_min",
            "t_max",
            "num_points",
            "raw_residual_mse",
            "raw_residual_rms",
            "rms_u",
            "rms_u_t",
            "rms_alpha_u_u_x",
            "rms_beta_u_xx",
            "rms_gamma_u_xxxx",
            "max_abs_u_xxxx",
        ]
        self.epochs_since_last_log = 0
        self.epochs_since_last_chunk = 0
        self.chunk_header_written = False
        self.eval_ready = False

    def on_train_begin(self):
        if self.log_every is None:
            self.log_every = self.model.display_every
        if self.chunk_every is None or self.chunk_every <= 0:
            self.chunk_every = 1000
        self.save_path = self.model.model_save_path + "/ks_diagnostics.txt"
        self.chunk_save_path = self.model.model_save_path + "/ks_time_chunks.csv"
        self.pde = self.model.pde
        self.model.ks_causal_chunk_diagnostics = None
        self.eval_ready = isinstance(self.pde, KuramotoSivashinskyEquation)
        if not self.eval_ready:
            return
        self._prepare_fixed_points()
        self.chunk_header_written = False

    @staticmethod
    def _rms(value):
        detached = value.detach()
        return torch.sqrt(torch.mean(detached ** 2))

    @staticmethod
    def _mse(value):
        detached = value.detach()
        return torch.mean(detached ** 2)

    @staticmethod
    def _slice_pde_points(model):
        x_train = getattr(model.train_state, "X_train", None)
        if x_train is None:
            return None
        bc_count = int(sum(getattr(model.data, "num_bcs", []) or []))
        if bc_count >= len(x_train):
            return None
        return x_train[bc_count:]

    @staticmethod
    def _derivative_column(value, points, column):
        if not value.requires_grad:
            return torch.zeros(
                (points.shape[0], 1),
                dtype=points.dtype,
                device=points.device,
            )
        grad = torch.autograd.grad(
            value,
            points,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )[0]
        if grad is None:
            return torch.zeros(
                (points.shape[0], 1),
                dtype=points.dtype,
                device=points.device,
            )
        return grad[:, column : column + 1]

    def _prepare_fixed_points(self):
        bbox = self.pde.bbox
        x_min, x_max, t_min, t_max = bbox
        x_grid = np.linspace(x_min, x_max, 128, dtype=np.float32)
        t_grid = np.linspace(t_min, t_max, 64, dtype=np.float32)
        xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
        self.grid_points = np.stack((xx.reshape(-1), tt.reshape(-1)), axis=1).astype(np.float32)

        if getattr(self.pde.geomtime, "random_points", None) is not None:
            self.validation_points = self.pde.geomtime.random_points(4096, random="pseudo").astype(np.float32)
        else:
            rng = np.random.default_rng(12345)
            self.validation_points = np.column_stack(
                (
                    rng.uniform(x_min, x_max, size=4096),
                    rng.uniform(t_min, t_max, size=4096),
                )
            ).astype(np.float32)

        x_ic = np.linspace(x_min, x_max, 512, dtype=np.float32)
        self.ic_points = np.column_stack((x_ic, np.full_like(x_ic, t_min))).astype(np.float32)

        boundary_t = np.linspace(t_min, t_max, 512, dtype=np.float32)
        self.left_boundary_points = np.column_stack(
            (np.full_like(boundary_t, x_min), boundary_t)
        ).astype(np.float32)
        self.right_boundary_points = np.column_stack(
            (np.full_like(boundary_t, x_max), boundary_t)
        ).astype(np.float32)

        ref_data = getattr(self.pde, "ref_data", None)
        self.ic_ref = None
        if ref_data is not None:
            nan_mask = np.isnan(ref_data).any(axis=1)
            clean_ref = ref_data[~nan_mask]
            if len(clean_ref) > 0:
                x_ref = clean_ref[:, : self.pde.input_dim]
                y_ref = clean_ref[:, self.pde.input_dim :]
                self._ref_interp = scipy.interpolate.NearestNDInterpolator(x_ref, y_ref)
                self.ic_ref = np.asarray(self._ref_interp(self.ic_points), dtype=np.float32)

        self.chunk_bounds = np.linspace(t_min, t_max, 17, dtype=np.float32)

    def _points_to_tensor(self, x_numpy):
        param = next(self.model.net.parameters())
        return torch.as_tensor(x_numpy, dtype=param.dtype, device=param.device).clone().detach().requires_grad_(True)

    def _evaluate_terms(self, x_numpy):
        with torch.enable_grad():
            was_training = self.model.net.training
            self.model.net.train(False)
            try:
                x = self._points_to_tensor(x_numpy)
                u = self.model.net(x)
                u_x = self._derivative_column(u, x, 0)
                u_t = self._derivative_column(u, x, 1)
                u_xx = self._derivative_column(u_x, x, 0)
                u_xxx = self._derivative_column(u_xx, x, 0)
                u_xxxx = self._derivative_column(u_xxx, x, 0)
                terms = self.pde.build_terms(u, u_t, u_x, u_xx, u_xxxx)
                terms["x"] = x
                terms["u_xxx"] = u_xxx
                return terms
            finally:
                self.model.net.train(was_training)

    def _basic_stats(self, terms):
        sum_term_rms = (
            self._rms(terms["term_t"])
            + self._rms(terms["term_adv"])
            + self._rms(terms["term_diff"])
            + self._rms(terms["term_hyper"])
        )
        abs_u4 = torch.abs(terms["u_xxxx"].detach()).reshape(-1)
        max_index = int(torch.argmax(abs_u4).item())
        return {
            "rms_u": float(self._rms(terms["u"]).detach().cpu().item()),
            "rms_u_t": float(self._rms(terms["u_t"]).detach().cpu().item()),
            "rms_alpha_u_u_x": float(self._rms(terms["term_adv"]).detach().cpu().item()),
            "rms_beta_u_xx": float(self._rms(terms["term_diff"]).detach().cpu().item()),
            "rms_gamma_u_xxxx": float(self._rms(terms["term_hyper"]).detach().cpu().item()),
            "rms_residual": float(self._rms(terms["residual"]).detach().cpu().item()),
            "cancellation_ratio": float(
                (self._rms(terms["residual"]) / (sum_term_rms + 1e-14)).detach().cpu().item()
            ),
            "p99_abs_u_xxxx": float(torch.quantile(abs_u4, 0.99).detach().cpu().item()),
            "max_abs_u_xxxx": float(abs_u4[max_index].detach().cpu().item()),
            "max_u_xxxx_x": float(terms["x"][max_index, 0].detach().cpu().item()),
            "max_u_xxxx_t": float(terms["x"][max_index, 1].detach().cpu().item()),
        }

    def _residual_mse(self, x_numpy):
        terms = self._evaluate_terms(x_numpy)
        return float(self._mse(terms["residual"]).detach().cpu().item())

    def _ic_mse(self):
        if self.ic_ref is None:
            return np.nan
        pred = self.model.predict(self.ic_points)
        return float(np.mean((pred - self.ic_ref) ** 2))

    def _periodic_gaps(self):
        left = self._evaluate_terms(self.left_boundary_points)
        right = self._evaluate_terms(self.right_boundary_points)
        return {
            "periodic_gap_u": float(self._rms(left["u"] - right["u"]).detach().cpu().item()),
            "periodic_gap_u_x": float(self._rms(left["u_x"] - right["u_x"]).detach().cpu().item()),
            "periodic_gap_u_xx": float(self._rms(left["u_xx"] - right["u_xx"]).detach().cpu().item()),
            "periodic_gap_u_xxx": float(self._rms(left["u_xxx"] - right["u_xxx"]).detach().cpu().item()),
        }

    def _global_stats(self, train_points):
        train_mse = self._residual_mse(train_points)
        validation_mse = self._residual_mse(self.validation_points)
        grid_terms = self._evaluate_terms(self.grid_points)
        stats = {
            "train_pde_mse": train_mse,
            "validation_pde_mse": validation_mse,
            "grid_pde_mse": float(self._mse(grid_terms["residual"]).detach().cpu().item()),
            "ic_mse": self._ic_mse(),
        }
        stats.update(self._basic_stats(grid_terms))
        stats.update(self._periodic_gaps())
        return stats, grid_terms

    def _write_chunk_rows(self, rows):
        array = np.asarray(rows, dtype=float)
        header = ",".join(self.chunk_keys)
        mode = "ab" if self.chunk_header_written else "wb"
        with open(self.chunk_save_path, mode) as handle:
            np.savetxt(handle, array, delimiter=",", header="" if self.chunk_header_written else header, comments="")
        self.chunk_header_written = True

    def _chunk_diagnostics(self, step, grid_terms):
        rows = []
        residual_rms = []
        for chunk_id in range(16):
            t_left = float(self.chunk_bounds[chunk_id])
            t_right = float(self.chunk_bounds[chunk_id + 1])
            if chunk_id == 15:
                mask = (self.grid_points[:, 1] >= t_left) & (self.grid_points[:, 1] <= t_right)
            else:
                mask = (self.grid_points[:, 1] >= t_left) & (self.grid_points[:, 1] < t_right)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            idx = torch.as_tensor(indices, device=grid_terms["u"].device, dtype=torch.long)
            chunk_terms = {key: value.index_select(0, idx) for key, value in grid_terms.items() if torch.is_tensor(value)}
            row = [
                step,
                chunk_id,
                t_left,
                t_right,
                len(indices),
                float(self._mse(chunk_terms["residual"]).detach().cpu().item()),
                float(self._rms(chunk_terms["residual"]).detach().cpu().item()),
                float(self._rms(chunk_terms["u"]).detach().cpu().item()),
                float(self._rms(chunk_terms["u_t"]).detach().cpu().item()),
                float(self._rms(chunk_terms["term_adv"]).detach().cpu().item()),
                float(self._rms(chunk_terms["term_diff"]).detach().cpu().item()),
                float(self._rms(chunk_terms["term_hyper"]).detach().cpu().item()),
                float(torch.max(torch.abs(chunk_terms["u_xxxx"].detach())).cpu().item()),
            ]
            rows.append(row)
            residual_rms.append(row[6])

        if rows:
            self._write_chunk_rows(rows)

        if self.verbose and rows:
            worst_idx = int(np.argmax(residual_rms))
            worst = rows[worst_idx]
            print(
                "KS chunk diagnostics: "
                f"step={step} "
                f"chunk_residual_rms_min={np.min(residual_rms):.10e} "
                f"chunk_residual_rms_mean={np.mean(residual_rms):.10e} "
                f"chunk_residual_rms_max={np.max(residual_rms):.10e} "
                f"worst_chunk_id={int(worst[1])} "
                f"worst_chunk_t_min={worst[2]:.10e} "
                f"worst_chunk_t_max={worst[3]:.10e}"
            )

    def _cache_causal_chunk_terms(self, step, grid_terms):
        details = getattr(self.model, "causal_loss_details", None)
        options = getattr(self.model, "causal_loss_options", None) or {}
        if not details or not options.get("enabled", False):
            self.model.ks_causal_chunk_diagnostics = None
            return
        num_chunks = int(details["chunk_losses"].numel())
        time_index = int(options.get("time_index", -1))
        if "t_min" in details and "t_max" in details:
            t_min = float(details["t_min"].reshape(-1)[0].detach().cpu())
            t_max = float(details["t_max"].reshape(-1)[-1].detach().cpu())
        else:
            pde = getattr(self, "pde", getattr(self.model, "pde", None))
            bbox = None if pde is None else np.asarray(pde.bbox, dtype=float)
            fallback_min = np.min(self.grid_points[:, time_index]) if bbox is None else bbox[-2]
            fallback_max = np.max(self.grid_points[:, time_index]) if bbox is None else bbox[-1]
            t_min = float(options.get("t_min", fallback_min))
            t_max = float(options.get("t_max", fallback_max))
        edges = np.linspace(t_min, t_max, num_chunks + 1)
        bin_ids = np.searchsorted(
            edges[1:-1], self.grid_points[:, time_index], side="right"
        )
        relative_eps = float(
            getattr(self.model, "dynamic_freezing_relative_eps", 1e-12)
        )
        chunks = {}
        for chunk_id in range(num_chunks):
            indices = np.flatnonzero(bin_ids == chunk_id)
            if len(indices) == 0:
                continue
            idx = torch.as_tensor(indices, device=grid_terms["u"].device, dtype=torch.long)
            values = {
                key: self._rms(grid_terms[key].index_select(0, idx))
                for key in ("term_t", "term_adv", "term_diff", "term_hyper", "residual")
            }
            scale = torch.sqrt(
                values["term_t"] ** 2
                + values["term_adv"] ** 2
                + values["term_diff"] ** 2
                + values["term_hyper"] ** 2
            )
            denominator = (
                values["term_t"]
                + values["term_adv"]
                + values["term_diff"]
                + values["term_hyper"]
                + relative_eps
            )
            chunks[chunk_id] = {
                "ut_rms": float(values["term_t"].detach().cpu().item()),
                "nonlinear_rms": float(values["term_adv"].detach().cpu().item()),
                "uxx_rms": float(values["term_diff"].detach().cpu().item()),
                "uxxxx_rms": float(values["term_hyper"].detach().cpu().item()),
                "ks_term_scale_rms": float(scale.detach().cpu().item()),
                "residual_to_term_scale_ratio": float(
                    (values["residual"] / denominator).detach().cpu().item()
                ),
            }
        self.model.ks_causal_chunk_diagnostics = {
            "step": int(step),
            "chunks": chunks,
        }

    def _save_heatmaps(self, step, grid_terms):
        x = self.grid_points[:, 0]
        t = self.grid_points[:, 1]
        residual = grid_terms["residual"].detach().cpu().numpy().reshape(-1)
        abs_u4 = torch.abs(grid_terms["u_xxxx"].detach()).cpu().numpy().reshape(-1)
        plot.plot_heatmap(
            x,
            t,
            residual,
            path=f"{self.model.model_save_path}/ks_residual_heatmap_step_{step}.png",
            title=f"KS residual step {step}",
            xlabel="x",
            ylabel="t",
            pde=self.pde,
        )
        plot.plot_heatmap(
            x,
            t,
            np.log10(1.0 + abs_u4),
            path=f"{self.model.model_save_path}/ks_log_u_xxxx_heatmap_step_{step}.png",
            title=f"KS log10(1 + abs(u_xxxx)) step {step}",
            xlabel="x",
            ylabel="t",
            pde=self.pde,
        )

    def on_epoch_end(self):
        if not self.eval_ready:
            return

        self.epochs_since_last_log += 1
        self.epochs_since_last_chunk += 1
        do_global = self.log_every is not None and self.epochs_since_last_log >= self.log_every
        do_chunk = self.chunk_every is not None and self.epochs_since_last_chunk >= self.chunk_every
        if not do_global and not do_chunk:
            return

        x_pde = self._slice_pde_points(self.model)
        if x_pde is None or len(x_pde) == 0:
            return

        step = self.model.train_state.step
        stats = None
        grid_terms = None
        if do_global or do_chunk:
            stats, grid_terms = self._global_stats(x_pde)

        if do_global:
            self.epochs_since_last_log = 0
            self._cache_causal_chunk_terms(step, grid_terms)
            row = [step] + [stats[key] for key in self.keys]
            self.rows.append(row)
            if self.verbose:
                summary = ", ".join(f"{key}={stats[key]:.10e}" for key in self.keys)
                print(f"KS diagnostics: step={step}, {summary}")

        if do_chunk:
            self.epochs_since_last_chunk = 0
            self._chunk_diagnostics(step, grid_terms)
            self._save_heatmaps(step, grid_terms)

    def on_train_end(self):
        if not self.rows:
            return
        header = "step, " + ", ".join(self.keys)
        np.savetxt(self.save_path, np.asarray(self.rows, dtype=float), header=header)


class TesterCallback(Callback):

    def __init__(
        self,
        log_every=100,
        verbose=True,
        fRMSE_param={'enable':True, 'iLow':5, 'iHigh':13, 'calc_every':2000},
        additional_metrics_fn=None,
    ):
        super(TesterCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fRMSE = fRMSE_param.get('enable', True)
        if self.fRMSE:
            self.fRMSE_l = fRMSE_param.get('iLow', 5)
            self.fRMSE_h = fRMSE_param.get('iHigh', 13)
            self.fRMSE_every = fRMSE_param.get('calc_every', 2000)

        self.indexes = []
        self.maes = []    # Mean Average Error
        self.mses = []    # Mean Square Error
        self.mxes = []    # Maximum Error
        self.l1res = []   # L1 Relative Error
        self.l2res = []   # L2 Relative Error
        self.crmses = []  # CSV_Loss
        self.frmses = []  # Mean Square Error in Fourier Space
        self.additional_metrics_fn = additional_metrics_fn
        self.additional_metrics = []

        self.ic_mses = []
        self.bc_mses = []
        self.bc_rmses = []
        self.bc_l2res = []

        self.mses_interp = []      # MSE на train_x, exact = nearest(ref_data)
        self.bc_mse_interp = []    # MSE на train_x_bc, exact = nearest(ref_data)


        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
        self.disable = False
        self._warned_missing_bc_ref = False
        self.test_x_bc = None
        self.test_y_bc = None

    def on_train_begin(self):
        self.save_path = self.model.model_save_path + "/"
        pde = self.model.pde

        # Load / Generate Test Data
        if pde.ref_sol is not None: # sample points from geometry
            sample_points = 2500 if pde.input_dim == 2 else 20000
            if getattr(self.model.data.geom, "uniform_points", None) is None:
                logger.warning(f"Method \'Uniform Points\' not found for class {type(self.model.data.geom)}, \
                                 Use random points for testing ...")
                sample_func = self.model.data.geom.random_points
            else:
                sample_func = self.model.data.geom.uniform_points
            
            self.test_x = sample_func(sample_points, boundary=True)
            self.test_y = pde.ref_sol(self.test_x)

            bc_sample_points = max(sample_points // 10, 1024)
            if getattr(self.model.data.geom, "uniform_boundary_points", None) is not None:
                self.test_x_bc = self.model.data.geom.uniform_boundary_points(bc_sample_points)
            elif getattr(self.model.data.geom, "random_boundary_points", None) is not None:
                self.test_x_bc = self.model.data.geom.random_boundary_points(bc_sample_points)
            if self.test_x_bc is not None:
                self.test_y_bc = pde.ref_sol(self.test_x_bc)
        elif pde.ref_data is not None:
            nan_mask = np.isnan(pde.ref_data).any(axis=1)
            self.test_x = pde.ref_data[~nan_mask, :pde.input_dim]
            self.test_y = pde.ref_data[~nan_mask, pde.input_dim:]
        else:
            self.disable = True
            logger.info("No reference solution or data provided, skipping TesterCallback")
            return
        
                # nearest exact(x) based on reference grid (like griddata(..., method="nearest"))
        # works for output_dim >= 1 too: values can be (N, out_dim)
        self._exact_near = scipy.interpolate.NearestNDInterpolator(self.test_x, self.test_y)

        self.solution_l1 = np.abs(self.test_y).mean()
        self.solution_l2 = np.sqrt((self.test_y**2).mean())

        # Для граничных условий
        eps = 1e-12
        X = self.test_x
        bbox = np.asarray(pde.bbox)  # len = 2*input_dim

        geom = self.model.data.geom
        has_time = isinstance(geom, dde.geometry.GeometryXTime) or isinstance(geom, dde.geometry.TimeDomain)

        # # предполагаем time_dim = последний, если задача time-dependent
        # has_time = (pde.input_dim >= 2)  # достаточно безопасно для твоих time-задач
        # time_dim = pde.input_dim - 1

        # IC: t == t_min (только если есть time)
        if has_time:
            time_dim = pde.input_dim - 1
            t_min = bbox[2 * time_dim]
            self.ic_mask = np.isclose(X[:, time_dim], t_min, atol=eps)
            
            t_vals = X[:, time_dim]
            # print("t range:", float(t_vals.min()), float(t_vals.max()), "unique-ish:", len(np.unique(t_vals[:min(10000,len(t_vals))])))

        else:
            self.ic_mask = np.zeros(len(X), dtype=bool)

        # BC: любая пространственная координата на min/max (исключая time dim), и не IC
        bc_mask = np.zeros(len(X), dtype=bool)
        spatial_dims = range(pde.input_dim - 1) if has_time else range(pde.input_dim)
        spatial_geom = getattr(geom, "geometry", geom) if has_time else geom

        if isinstance(spatial_geom, dde.geometry.Hypersphere):
            center = np.asarray(spatial_geom.center)
            radius = float(spatial_geom.radius)
            bc_mask = np.isclose(
                np.linalg.norm(X[:, list(spatial_dims)] - center, axis=1),
                radius,
                atol=1e-6,
            )
        else:
            for d in spatial_dims:
                lo = bbox[2 * d]
                hi = bbox[2 * d + 1]
                bc_mask |= np.isclose(X[:, d], lo, atol=eps) | np.isclose(X[:, d], hi, atol=eps)

        self.bc_mask = bc_mask & (~self.ic_mask)  # “только BC, без IC”

        if not np.any(self.bc_mask) and pde.ref_data is not None and not self._warned_missing_bc_ref:
            logger.warning(
                "TesterCallback found no reference points on the spatial boundary for %s. "
                "Boundary RMSE on the reference grid is undefined and will stay NaN. "
                "Reference bbox: %s.",
                type(pde).__name__,
                pde.bbox,
            )
            self._warned_missing_bc_ref = True

        if self.fRMSE:
            self.frmse_init()

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        self.valid_epoch += 1
        if self.disable or self.log_every is None or self.epochs_since_last_resample < self.log_every:
            return
        self.epochs_since_last_resample = 0

        with torch.no_grad():
            y = self.model.predict(self.test_x)

        mse = ((y - self.test_y)**2).mean()
        mae = np.abs(y - self.test_y).mean()
        mxe = np.max(np.abs(y - self.test_y))
        l1re = mae / self.solution_l1
        l2re = np.sqrt(mse) / self.solution_l2
        crmse = np.abs((y - self.test_y).mean())
        if self.fRMSE and self.valid_epoch % self.fRMSE_every == 0:
            frmse = self.frmse_calc(y)
        else:
            frmse = (np.nan, np.nan, np.nan)

        # IC MSE (на ref grid)
        if np.any(self.ic_mask):
            y_ic = self.model.predict(self.test_x[self.ic_mask])
            ic_mse = ((y_ic - self.test_y[self.ic_mask]) ** 2).mean()
        else:
            ic_mse = np.nan

        # BC MSE (на ref grid)
        if self.test_x_bc is not None and len(self.test_x_bc) > 0:
            y_bc = self.model.predict(self.test_x_bc)
            bc_mse = ((y_bc - self.test_y_bc) ** 2).mean()
            bc_rmse = np.sqrt(bc_mse)
            bc_l2re = bc_rmse / (self.solution_l2 + 1e-12)
        elif np.any(self.bc_mask):
            y_bc = self.model.predict(self.test_x[self.bc_mask])
            bc_mse = ((y_bc - self.test_y[self.bc_mask]) ** 2).mean()
            bc_rmse = np.sqrt(bc_mse)
            bc_l2re = bc_rmse / (self.solution_l2 + 1e-12)
        else:
            bc_mse = np.nan
            bc_rmse = np.nan
            bc_l2re = np.nan


        # --- Interpolation-based metrics (nearest exact on arbitrary grids) ---
        # 1) interp MSE on training points (prefer train_x; fallback to train_x_all)
        train_x = getattr(self.model.data, "train_x", None)
        if train_x is None:
            train_x = getattr(self.model.data, "train_x_all", None)

        if train_x is not None and len(train_x) > 0:
            y_train = self.model.predict(train_x)
            y_train_true = self._exact_near(train_x)
            mse_interp = ((y_train - y_train_true) ** 2).mean()
        else:
            mse_interp = np.nan

        # 2) interp BC MSE on DeepXDE BC collocation points (train_x_bc)
        bnd_x = getattr(self.model.data, "train_x_bc", None)
        if bnd_x is None:
            # если вдруг не посчитано — попробуем получить через data.bc_points()
            bc_points_fn = getattr(self.model.data, "bc_points", None)
            if callable(bc_points_fn):
                bnd_x = bc_points_fn()

        if bnd_x is not None and len(bnd_x) > 0:
            y_bnd = self.model.predict(bnd_x)
            y_bnd_true = self._exact_near(bnd_x)
            bc_mse_interp = ((y_bnd - y_bnd_true) ** 2).mean()
        else:
            bc_mse_interp = np.nan


        self.mses_interp.append(mse_interp)
        self.bc_mse_interp.append(bc_mse_interp)

        self.bc_mses.append(bc_mse)
        self.bc_rmses.append(bc_rmse)
        self.bc_l2res.append(bc_l2re)

        self.ic_mses.append(ic_mse)

        self.indexes.append(self.valid_epoch)
        self.mses.append(mse)
        self.maes.append(mae)
        self.mxes.append(mxe)
        self.l1res.append(l1re)
        self.l2res.append(l2re)
        self.crmses.append(crmse)
        self.frmses.append(frmse)

        if self.additional_metrics_fn is not None:
            extra_metrics = {
                str(name): float(value)
                for name, value in self.additional_metrics_fn(y, self.test_y).items()
            }
            self.additional_metrics.append(extra_metrics)
        else:
            extra_metrics = None


        if self.verbose:
            if np.isnan(frmse[0]):
                print('Validation: epoch {} MSE {:.10e} MAE {:.10e} MXE {:.10e} BMSE {:.10e} ICMSE {:.10e} L1RE {:.10e} L2RE {:.10e} CRMSE {:.10e}'.\
                       format(self.valid_epoch, mse, mae, mxe, bc_mse, ic_mse, l1re, l2re, crmse))
            else:
                print('Validation: epoch {} MSE {:.10e} MAE {:.10e} MXE {:.10e} BMSE {:.10e} ICMSE {:.10e} L1RE {:.10e} L2RE {:.10e} CRMSE {:.10e} FRMSE ({:.10e}, {:.10e}, {:.10e})'.\
                       format(self.valid_epoch, mse, mae, mxe, bc_mse, ic_mse, l1re, l2re, crmse, frmse[0], frmse[1], frmse[2]))
            if extra_metrics:
                summary = " ".join(
                    f"{name} {value:.10e}" for name, value in extra_metrics.items()
                )
                print(f"Long-horizon metrics: epoch {self.valid_epoch} {summary}")

    def on_train_end(self):
        if self.disable:
            return

        self.indexes = np.array(self.indexes)
        self.frmses = np.asarray(self.frmses, dtype=float)
        if self.frmses.size == 0:
            self.frmses = np.empty((0, 3), dtype=float)
        elif self.frmses.ndim == 1:
            if self.frmses.shape[0] == len(self.indexes) * 3 and len(self.indexes) > 0:
                self.frmses = self.frmses.reshape(len(self.indexes), 3)
            elif self.frmses.shape[0] == len(self.indexes):
                self.frmses = np.repeat(self.frmses[:, None], 3, axis=1)
            elif self.frmses.shape[0] == 3:
                self.frmses = np.repeat(self.frmses[None, :], len(self.indexes), axis=0)
            else:
                self.frmses = np.full((len(self.indexes), 3), np.nan, dtype=float)
        np.savetxt(
            self.save_path + 'errors.txt',
            np.array([self.indexes, self.maes, self.mses, self.mxes, self.bc_mses, self.l1res, self.l2res, self.crmses,\
                      self.frmses[:, 0], self.frmses[:, 1], self.frmses[:, 2], self.mses_interp, self.bc_mse_interp]).T,
            header="epochs, maes, mses, mxes, bnd_mse, l1res, l2res, crmses, frmses(low, mid, high), mses_interp, bc_mse_interp"
        )
        if self.additional_metrics:
            metric_names = list(self.additional_metrics[0])
            rows = np.asarray(
                [
                    [epoch] + [values.get(name, np.nan) for name in metric_names]
                    for epoch, values in zip(self.indexes, self.additional_metrics)
                ],
                dtype=float,
            )
            np.savetxt(
                self.save_path + 'long_horizon_metrics.txt',
                rows,
                header="epochs, " + ", ".join(metric_names),
            )

        plot.plot_lines([self.indexes, self.maes], xlabel="epochs", labels=['maes'], path=self.save_path + "maes.png", title="mean average error")
        plot.plot_lines([self.indexes, self.mses], xlabel="epochs", labels=['mses'], path=self.save_path + "mses.png", title="mean square error")
        plot.plot_lines([self.indexes, self.mxes], xlabel="epochs", labels=['mxes'], path=self.save_path + "mxes.png", title="maximum error")
        plot.plot_lines([self.indexes, self.ic_mses], xlabel="epochs", labels=['ic_mses'],
                path=self.save_path + "ic_mses.png", title="IC mean square error (ref grid)")

        plot.plot_lines([self.indexes, self.bc_mses], xlabel="epochs", labels=['bc_mses'],
                        path=self.save_path + "bc_mses.png", title="BC mean square error (ref grid)")
        
        plot.plot_lines([self.indexes, self.bc_rmses], xlabel="epochs", labels=['bc_mses'],
                        path=self.save_path + "bc_rmses.png", title="BC root mean square error (ref grid)")
        
        plot.plot_lines([self.indexes, self.bc_l2res], xlabel="epochs", labels=['bc_mses'],
                        path=self.save_path + "bc_l2res.png", title="BC l2re error (ref grid)")
        
        plot.plot_lines([self.indexes, self.mses_interp],
                xlabel="epochs", labels=['mses_interp'],
                path=self.save_path + "mses_interp.png",
                title="MSE on train grid (nearest exact)")

        plot.plot_lines([self.indexes, self.bc_mse_interp],
                        xlabel="epochs", labels=['bc_mse_interp'],
                        path=self.save_path + "bc_mse_interp.png",
                        title="BC MSE on train_x_bc (nearest exact)")
        
        plot.plot_lines([self.indexes, self.l1res, self.l2res],
                        xlabel="epochs",
                        labels=['l1re', 'l2re'],
                        path=self.save_path + "relerr.png",
                        title="relative error")
        X = ~np.isnan(self.frmses).any(axis=1)
        plot.plot_lines([self.indexes[X], self.frmses[X, 0], self.frmses[X, 1], self.frmses[X, 2]], 
                        xlabel="epochs", 
                        labels=['low freq', 'mid freq', 'high freq'], 
                        path=self.save_path + "frmses.png", 
                        title="mean square error in fourier space")
        
        self.rmse = np.sqrt(self.mses[-1])
        self.brmse = self.bc_rmses[-1]

        self.indexes = []
        self.maes = []   
        self.mses = []   
        self.mxes = []   
        self.l1res = []  
        self.l2res = []  
        self.crmses = [] 
        self.frmses = [] 
        self.additional_metrics = []

        self.ic_mses = []
        self.bc_mses = []
        self.bc_rmses = []
        self.bc_l2res = []

        self.mses_interp = []   
        self.bc_mse_interp = []   

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
    
    def frmse_init(self):
        pde = self.model.pde
        if not isinstance(pde.geom, Hypercube) and not isinstance(pde.geom, Interval):
            logger.warning(f"Fourier transform errors are enabled only in Interval / Hypercube and their combination with Time domains. \
                           Type {type(pde.geom).__name__} is not a valid geometry and fRMSE has been disabled")
            self.fRMSE=False
            return
        if pde.input_dim > 3:
            logger.warning(f"For high dimensional PDEs like {type(pde).__name__} with dim {pde.input_dim} is slow to calculate fRMSE. \
                           fRMSE has been disabled")
            self.fRMSE=False
            return 

        # prepare calculation
        self.test_x_delaunay = scipy.spatial.Delaunay(self.test_x)
        ptn = 3e4 # generate about 3e4 uniform sampling points in the domain
        for i in range(pde.input_dim):
            ptn /= pde.bbox[i * 2 + 1] - pde.bbox[i * 2]
        ptn = ptn ** (1 / pde.input_dim)
        xlist = [np.linspace(pde.bbox[i * 2], pde.bbox[i * 2 + 1], int(np.ceil((pde.bbox[i*2+1] - pde.bbox[i*2]) * ptn)) + 1, endpoint=False)[1:] \
                 for i in range(pde.input_dim)]
        self.sample_x = np.stack(np.meshgrid(*xlist), axis=-1)
    
    def frmse_calc(self, y):
        pde = self.model.pde
        res = scipy.interpolate.LinearNDInterpolator(self.test_x_delaunay, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        resn = scipy.interpolate.NearestNDInterpolator(self.test_x, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        res[np.isnan(res)] = resn[np.isnan(res)]
        err = np.fft.rfftn(res, axes=tuple(range(res.ndim-1))) # transform except the last dim (pde.output_dim)
        err = np.mean(np.abs(err) ** 2 / res.size, axis=-1) # take average through the last dim

        if pde.input_dim == 1:
            err_low = err[:self.fRMSE_l].mean()
            err_mid = err[self.fRMSE_l:self.fRMSE_h].mean()
            err_high = err[self.fRMSE_h:].mean()
        else:
            err_low, err_mid, err_high = 0.0, 0.0, 0.0
            err_low_cnt, err_mid_cnt, err_high_cnt = 0, 0, 0
            for ids in itertools.product(*[range((k+1)//2) for k in err.shape[:-1]]):
                freq2 = sum(i ** 2 for i in ids)
                ilow = min(int(np.sqrt(max(0, self.fRMSE_l**2 - freq2))), err.shape[-1])
                ihigh = min(int(np.sqrt(max(0, self.fRMSE_h**2 - freq2))), err.shape[-1])

                err_low += err[(*ids, slice(None, ilow, None))].sum()
                err_mid += err[(*ids, slice(ilow, ihigh, None))].sum()
                err_high += err[(*ids, slice(ihigh, None, None))].sum()

                err_low_cnt += ilow 
                err_mid_cnt += ihigh - ilow
                err_high_cnt += err.shape[-1] - ihigh
            
            err_low /= err_low_cnt # calculate mean square error
            err_mid /= err_mid_cnt
            err_high /= err_high_cnt

        return err_low, err_mid, err_high
    

class ModelSaverCallback(Callback):
    def __init__(self, total_iterations, n_save_models=10):
        super(ModelSaverCallback, self).__init__()
        self.total_iterations = total_iterations
        self.n_save_models = n_save_models
        # Вычисляем интервал сохранения (чтобы сохранить ровно n_save_models моделей)
        self.save_every = max(1, total_iterations // n_save_models)
        self.saved_models = []  # здесь будут храниться копии моделей
        self.next_save_iter = self.save_every  # первое сохранение после save_every итераций

    def on_epoch_end(self):
        # Проверяем, что модель скомпилирована и есть доступ к номеру итерации
        # if not hasattr(self, 'model') or self.model.train_state is None:
        #     return
        current_iter = self.model.train_state.step

        # Если достигли очередного рубежа сохранения
        if current_iter >= self.next_save_iter and len(self.saved_models) < self.n_save_models:
            # Делаем глубокую копию модели (только сеть, так как весь объект model может быть сложным)
            model_copy = copy.deepcopy(self.model.net)
            self.saved_models.append(model_copy)
            print(f"Model saved at iteration {current_iter} ({len(self.saved_models)}/{self.n_save_models})")
            # Устанавливаем следующий рубеж
            self.next_save_iter += self.save_every

    def on_train_end(self):
        # Если сохранили меньше, чем планировали (например, обучение рано остановилось), можно добавить последнюю модель
        if len(self.saved_models) < self.n_save_models and hasattr(self, 'model'):
            model_copy = copy.deepcopy(self.model.net)
            self.saved_models.append(model_copy)
            print(f"Final model added at end of training ({len(self.saved_models)}/{self.n_save_models})")

        self.model.train_state.epoch = 0 
        self.model.train_state.step = 0
