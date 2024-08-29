"""Module for plotting the results of the experiments."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np
from sicore import SummaryFigure, rejection_rate  # type: ignore[import]

from experiment.utils import Results


def plot_main(option: str, mode: str) -> None:
    """Plot the results of the experiments."""
    values: list[float]
    ylabel, is_null, num_seeds = "Type I Error Rate", True, 9
    fig_path = Path("figures/main") / f"fpr_{option}_{mode}.pdf"
    match mode:
        case "n":
            values = [100, 200, 300, 400]
            result_name = lambda value, seed: f"{value}_20_0.0_{seed}.pkl"
            xlabel = "number of samples"
        case "d":
            values = [10, 20, 30, 40]
            result_name = lambda value, seed: f"200_{value}_0.0_{seed}.pkl"
            xlabel = "number of features"
        case "delta":
            values = [0.2, 0.4, 0.6, 0.8]
            result_name = lambda value, seed: f"200_20_{value}_{seed}.pkl"
            fig_path = Path("figures/main") / f"tpr_{option}.pdf"
            xlabel, ylabel, is_null, num_seeds = "signal", "Power", False, 1

    figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
    for value in values:
        results = Results()
        for seed in range(num_seeds):
            path = Path(f"results_{option}") / result_name(value, seed)
            with path.open("rb") as f:
                results += pickle.load(f)
        assert len(results.p_values) == num_seeds * 1000

        figure.add_results(results.p_values, label="proposed", xloc=value)
        figure.add_results(results.oc_p_values, label="oc", xloc=value)
        if is_null:
            figure.add_results(results.naive_p_values, label="naive", xloc=value)

    if is_null:
        figure.add_red_line(value=0.05, label="significance level")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(fig_path, fontsize=16, legend_loc="upper left")


def plot_time(mode: str) -> None:
    """Plot the computation time of the experiments."""
    match mode:
        case "n":
            values = [100, 200, 300, 400]
            result_name = lambda value, seed: f"{value}_20_0.0_{seed}.pkl"
            fig_path = Path("figures/time") / "time_n.pdf"
            xlabel = "number of samples"
        case "d":
            values = [10, 20, 30, 40]
            result_name = lambda value, seed: f"200_{value}_0.0_{seed}.pkl"
            fig_path = Path("figures/time") / "time_d.pdf"
            xlabel = "number of features"

    num_seeds = 1
    figure = SummaryFigure(xlabel=xlabel, ylabel="Computation Time (s)")

    for value in values:
        for option in ["op1", "op2", "all_cv"]:
            results = Results()
            for seed in range(num_seeds):
                dir_path = Path(f"results_{option}")
                path = dir_path / result_name(value, seed)
                with path.open("rb") as f:
                    results += pickle.load(f)
            assert len(results.times) == num_seeds * 1000

            figure.add_value(np.mean(results.times).item(), xloc=value, label=option)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(fig_path, fontsize=16, ylim=None)


def plot_real(option: str, key: str) -> None:
    """Plot the results of the real data experiments."""
    figure = SummaryFigure(xlabel="number of samples", ylabel="Power")
    num_seeds = 1

    for n in [100, 150, 200]:
        results = Results()
        for seed in range(num_seeds):
            path = Path("results_real") / f"{option}_{key}_{n}_{seed}.pkl"
            with path.open("rb") as f:
                results += pickle.load(f)
        assert len(results.p_values) == num_seeds * 1000

        figure.add_results(results.p_values, label="proposed", xloc=n)
        figure.add_results(results.oc_p_values, label="oc", xloc=n)

    fig_path = Path("figures/real") / f"{option}_{key}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(fig_path, fontsize=16, legend_loc="upper left")


def print_real(option: str) -> None:
    """Print the results of the real data experiments."""
    keys = [
        "heating_load",
        "cooling_load",
        "gas_turbine",
        "red_wine",
        "white_wine",
        "abalone",
        "concrete",
        "housing",
    ]
    num_seeds = 1
    strings_list = []
    for n in [100, 150, 200]:
        string = f"$n={n}$"
        for key in keys:
            results = Results()
            for seed in range(num_seeds):
                path = Path("results_real") / f"{option}_{key}_{n}_{seed}.pkl"
                with path.open("rb") as f:
                    results += pickle.load(f)
            assert len(results.p_values) == num_seeds * 1000

            tpr_ = rejection_rate(results.p_values, alpha=0.05)
            oc_tpr_ = rejection_rate(results.oc_p_values, alpha=0.05)
            tpr = Decimal(str(tpr_)).quantize(
                Decimal("0.01"),
                "ROUND_HALF_UP",
            )
            oc_tpr = Decimal(str(oc_tpr_)).quantize(
                Decimal("0.01"),
                "ROUND_HALF_UP",
            )
            string += rf" & \textbf{{{str(tpr)[1:]}}}/{str(oc_tpr)[1:]}"
        strings_list.append(string)

    strings_path = Path(f"figures/real/{option}.txt")
    strings_path.parent.mkdir(parents=True, exist_ok=True)
    with strings_path.open("w") as f:
        f.write("\n".join(strings_list))


def plot_robust_non_gaussian(option: str, alpha: float = 0.05) -> None:
    """Plot the results of the robustness experiments for non-Gaussian noise."""
    figure = SummaryFigure(xlabel="Wasserstein Distance", ylabel="Type I Error Rate")
    num_seeds = 1

    for rv_name in ["skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"]:
        for distance in [0.01, 0.02, 0.03, 0.04]:
            results = Results()
            for seed in range(num_seeds):
                path = (
                    Path(f"results_robust/{rv_name}")
                    / f"{option}_{distance}_{seed}.pkl"
                )
                with path.open("rb") as f:
                    results += pickle.load(f)
            assert len(results.p_values) == num_seeds * 1000
            figure.add_results(
                results.p_values,
                label=rv_name,
                xloc=distance,
                alpha=alpha,
            )
    figure.add_red_line(value=alpha, label="significance level")

    default_alpha = 0.05

    fig_path = Path("figures/robust") / f"{option}_non_gaussian_{alpha:.2f}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(
        fig_path,
        fontsize=14,
        legend_loc="upper left",
        ylim=(0.0, 0.2) if alpha == default_alpha else (0.0, 0.04),
        yticks=[0.0, 0.05, 0.10, 0.15, 0.2]
        if alpha == default_alpha
        else [0.0, 0.01, 0.02, 0.03, 0.04],
    )


def plot_robust_estimated(option: str, mode: str) -> None:
    """Plot the results of the robustness experiments for estimated variance."""
    match mode:
        case "n":
            values = [100, 200, 300]
            result_name = lambda value, seed: f"{option}_{value}_20_{seed}.pkl"
            fig_path = Path("figures/robust") / f"{option}_estimated_n.pdf"
            xlabel = "number of samples"
        case "d":
            values = [10, 20, 30, 40]
            result_name = lambda value, seed: f"{option}_200_{value}_{seed}.pkl"
            xlabel = "number of features"

    num_seeds = 1
    figure = SummaryFigure(xlabel=xlabel, ylabel="Type I Error Rate")
    for value in values:
        results = Results()
        for seed in range(num_seeds):
            path = Path("results_robust/estimated") / result_name(value, seed)
            with path.open("rb") as f:
                results += pickle.load(f)
        assert len(results.p_values) == num_seeds * 1000

        for alpha in [0.05, 0.01, 0.10]:
            figure.add_results(
                results.p_values,
                label=f"alpha={alpha:.2f}",
                xloc=value,
                alpha=alpha,
            )

    figure.add_red_line(value=0.05, label="significance levels")
    figure.add_red_line(value=0.01)
    figure.add_red_line(value=0.10)

    fig_path = Path("figures/robust") / f"{option}_estimated_{mode}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(
        fig_path,
        fontsize=16,
        legend_loc="upper left",
        ylim=(0.0, 0.2),
        yticks=[0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=32) as executor:
        for arg in product(["op1", "op2", "all_cv"], ["n", "d", "delta"]):
            executor.submit(plot_main, *arg)
        for mode in ["n", "d"]:
            executor.submit(plot_time, mode)
        for arg in product(
            ["all_cv"],
            [
                "heating_load",
                "cooling_load",
                "gas_turbine",
                "red_wine",
                "white_wine",
                "abalone",
                "concrete",
                "housing",
            ],
        ):
            executor.submit(plot_real, *arg)
        executor.submit(print_real, "all_cv")
        for arg_ in [("all_cv", 0.05), ("all_cv", 0.01)]:
            executor.submit(plot_robust_non_gaussian, *arg_)
        for arg in product(["all_cv"], ["n", "d"]):
            executor.submit(plot_robust_estimated, *arg)
