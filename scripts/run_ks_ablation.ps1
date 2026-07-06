param(
    [int]$Iterations = 40000,
    [string]$Optimizer = "soap",
    [string]$Out = "runs_ks_ablation"
)

$Runner = "experiments/Chaotic/run_chaotic.py"
$Common = @(
    $Runner,
    "--equation", "ks",
    "--iterations", "$Iterations",
    "--optimizer", "$Optimizer",
    "--out", $Out
)

function Run-Experiment {
    param(
        [string]$Name,
        [string[]]$ExtraArgs
    )

    Write-Host "Running $Name"
    $Args = $Common + @("--name", $Name) + $ExtraArgs
    python @Args
}

Run-Experiment "E0_baseline" @()

Run-Experiment "E1_causal" @(
    "--use-causal-loss",
    "--causal-num-chunks", "16",
    "--causal-tol", "0.1"
)

Run-Experiment "E2_fourier" @(
    "--use-fourier-features",
    "--fourier-num-modes-x", "16"
)

Run-Experiment "E3_resampling" @(
    "--resample-collocation",
    "--resample-every", "1"
)

Run-Experiment "E4_causal_fourier" @(
    "--use-causal-loss",
    "--causal-num-chunks", "16",
    "--causal-tol", "0.1",
    "--use-fourier-features",
    "--fourier-num-modes-x", "16"
)

Run-Experiment "E5_causal_resampling" @(
    "--use-causal-loss",
    "--causal-num-chunks", "16",
    "--causal-tol", "0.1",
    "--resample-collocation",
    "--resample-every", "1"
)

Run-Experiment "E6_fourier_resampling" @(
    "--use-fourier-features",
    "--fourier-num-modes-x", "16",
    "--resample-collocation",
    "--resample-every", "1"
)

Run-Experiment "E7_causal_fourier_resampling" @(
    "--use-causal-loss",
    "--causal-num-chunks", "16",
    "--causal-tol", "0.1",
    "--use-fourier-features",
    "--fourier-num-modes-x", "16",
    "--resample-collocation",
    "--resample-every", "1"
)

Run-Experiment "E8_all_windows" @(
    "--use-causal-loss",
    "--causal-num-chunks", "16",
    "--causal-tol", "0.1",
    "--use-fourier-features",
    "--fourier-num-modes-x", "16",
    "--resample-collocation",
    "--resample-every", "1",
    "--use-windows",
    "--num-windows", "10"
)
