"""Phase-insensitive long-horizon metrics for KS solution fields."""

import math

import numpy as np


def long_horizon_metrics(prediction, exact, late_fraction: float = 0.5):
    """Compare late-time energy, spectra, and empirical value distributions.

    ``prediction`` and ``exact`` must both have shape ``[time, space]``.
    Energy and spectra are computed after removing the spatial mean at each
    time level.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if prediction.shape != exact.shape or prediction.ndim != 2:
        raise ValueError("prediction and exact must have the same [time, space] shape")
    if not 0.0 < float(late_fraction) <= 1.0:
        raise ValueError("late_fraction must satisfy 0 < late_fraction <= 1")
    if prediction.size == 0:
        raise ValueError("prediction and exact must not be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(exact).all():
        return {
            "late_energy_agreement": float("nan"),
            "late_spectral_overlap": float("nan"),
            "late_normalized_wasserstein": float("nan"),
        }

    late_count = max(1, int(math.ceil(prediction.shape[0] * float(late_fraction))))
    pred_late = prediction[-late_count:]
    exact_late = exact[-late_count:]
    pred_centered = pred_late - pred_late.mean(axis=1, keepdims=True)
    exact_centered = exact_late - exact_late.mean(axis=1, keepdims=True)

    pred_energy = float(np.median(np.mean(pred_centered**2, axis=1)))
    exact_energy = float(np.median(np.mean(exact_centered**2, axis=1)))
    energy_floor = np.finfo(np.float64).eps * max(pred_energy, exact_energy, 1.0)
    if pred_energy <= energy_floor and exact_energy <= energy_floor:
        energy_agreement = 1.0
    else:
        energy_agreement = (
            2.0
            * pred_energy
            * exact_energy
            / (pred_energy**2 + exact_energy**2 + energy_floor**2)
        )

    pred_power = np.mean(np.abs(np.fft.rfft(pred_centered, axis=1)) ** 2, axis=0)
    exact_power = np.mean(np.abs(np.fft.rfft(exact_centered, axis=1)) ** 2, axis=0)
    spectral_denominator = float(pred_power.sum() + exact_power.sum())
    if spectral_denominator <= energy_floor:
        spectral_overlap = 1.0
    else:
        spectral_overlap = float(
            2.0 * np.minimum(pred_power, exact_power).sum() / spectral_denominator
        )

    pred_sorted = np.sort(pred_late.reshape(-1))
    exact_sorted = np.sort(exact_late.reshape(-1))
    wasserstein = float(np.mean(np.abs(pred_sorted - exact_sorted)))
    exact_scale = float(np.std(exact_late))
    normalized_wasserstein = wasserstein / max(
        exact_scale, np.finfo(np.float64).eps
    )
    return {
        "late_energy_agreement": float(np.clip(energy_agreement, 0.0, 1.0)),
        "late_spectral_overlap": float(np.clip(spectral_overlap, 0.0, 1.0)),
        "late_normalized_wasserstein": normalized_wasserstein,
    }
