"""Module for plotting the results of the experiments."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sicore import SelectiveInferenceResult, rejection_rate  # type: ignore[import]

from experiment.utils import Results


class SummaryFigure:
    """A class plotting a summary figure of experiments.

    Args:
        title (str | None, optional): Title of the figure. Defaults to None.
        xlabel (str | None, optional): Label of x-axis. Defaults to None.
        ylabel (str | None, optional): Label of y-axis. Defaults to None.
    """

    def __init__(
        self,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        """Initialize a summary figure.

        Args:
            title (str | None, optional): Title of the figure. Defaults to None.
            xlabel (str | None, optional): Label of x-axis. Defaults to None.
            ylabel (str | None, optional): Label of y-axis. Defaults to None.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data: dict[str, list] = {}

    def add_value(self, value: float, label: str, xloc: str | float) -> None:
        """Add a value to the figure.

        Args:
            value (float): Value to be plotted.
            label (str): Label corresponding to the value.
                To note that the label well be shown in the given order.
            xloc (str | float): Location of the value.
                If str, it will be equally spaced in the given order.
                If float, it will be the exact location.
        """
        self.data.setdefault(label, [])
        self.data[label].append((xloc, value))
        self.red_lines: list[tuple[float, str | None]] = []

    def add_results(
        self,
        results: list[SelectiveInferenceResult] | list[float] | np.ndarray,
        label: str,
        xloc: str | float,
        alpha: float = 0.05,
        *,
        naive: bool = False,
        bonferroni: bool = False,
        log_num_comparisons: float = 0.0,
    ) -> None:
        """Add rejection rate computed from the given results to the figure.

        Args:
            results (list[SelectiveInferenceResult] | list[float] | np.ndarray):
                List of SelectiveInferenceResult objects or p-values.
            label (str):
                Label corresponding to the results.
            xloc (str | float):
                Location of the results.
            alpha (float, optional): Significance level. Defaults to 0.05.
            naive (bool, optional):
                Whether to compute rejection rate of naive inference.
                This option is available only when results are
                SelectiveInferenceResult objects. Defaults to False.
            bonferroni (bool, optional):
                Whether to compute rejection rate with Bonferroni correction.
                This option is available only when results are
                SelectiveInferenceResult objects. Defaults to False.
            log_num_comparisons (float, optional):
                Logarithm of the number of comparisons for the Bonferroni correction.
                This option is ignored when bonferroni is False.
                Defaults to 0.0, which means no correction.
        """
        value = rejection_rate(
            results,
            alpha=alpha,
            naive=naive,
            bonferroni=bonferroni,
            log_num_comparisons=log_num_comparisons,
        )
        self.add_value(value, label, xloc)

    def add_red_line(self, value: float = 0.05, label: str | None = None) -> None:
        """Add a red line at the specified value.

        Args:
            value (float): Value to be plotted as a red line. Defaults to 0.05.
            label (str | None): Label of the red line. Defaults to None.
        """
        self.red_lines.append((value, label))

    def plot(
        self,
        filepath: Path | str | None = None,
        ylim: tuple[float, float] | None = (0.0, 1.0),
        yticks: list[float] | None = None,
        legend_loc: str | None = None,
        fontsize: int = 10,
    ) -> None:
        """Plot the figure.

        Args:
            filepath (Path | str | None, optional):
                File path. If `filepath` is given, the plotted figure
                will be saved as a file. Defaults to None.
            ylim (tuple[float, float] | None, optional):
                Range of y-axis. Defaults to None.
                If None, range of y-axis will be automatically determined.
            yticks (list[float] | None, optional):
                List of y-ticks. Defaults to None.
                If None, y-ticks will be automatically determined.
            legend_loc (str | None, optional):
                Location of the legend. Defaults to None.
                If None, the legend will be placed at the best location.
            fontsize (int, optional):
                Font size of the legend. Defaults to 10.
        """
        plt.rcParams.update({"font.size": fontsize})
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

        for label, xloc_value_list in self.data.items():
            xlocs_, values_ = zip(*xloc_value_list, strict=True)
            xlocs, values = np.array(xlocs_), np.array(values_)
            if not all(isinstance(xloc, (str)) for xloc in xlocs):
                values = values[np.argsort(xlocs)]
                xlocs = np.sort(xlocs)
            plt.plot(xlocs, values, label=label, marker="x")

        for value, label_ in self.red_lines:
            plt.plot(
                xlocs,
                [value] * len(xlocs),
                color="red",
                linestyle="--",
                lw=0.5,
                label=label_,
            )
        plt.xticks(xlocs)

        if ylim is not None:
            plt.ylim(ylim)
        if yticks is not None:
            plt.yticks(yticks)

        plt.legend(frameon=False, loc=legend_loc)
        if filepath is None:
            plt.show()
        else:
            filename = str(filepath) if isinstance(filepath, Path) else filepath
            plt.savefig(filename, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.close()


def plot_main(option: str, mode: str) -> None:
    """Plot the results of the experiments."""
    values: list[float]
    match mode:
        case "n":
            values = [100, 200, 300, 400]
            result_name = lambda value, seed: f"{value}_20_0.0_{seed}.pkl"  # noqa: E731
            fig_path = Path("figures/main") / f"fpr_{option}_n.pdf"
            xlabel = "number of samples"
            ylabel = "Type I Error Rate"
            is_null = True
            num_seeds = 1
        case "d":
            values = [10, 20, 30, 40]
            result_name = lambda value, seed: f"200_{value}_0.0_{seed}.pkl"  # noqa: E731
            fig_path = Path("figures/main") / f"fpr_{option}_d.pdf"
            xlabel = "number of features"
            ylabel = "Type I Error Rate"
            is_null = True
            num_seeds = 1
        case "delta":
            values = [0.2, 0.4, 0.6, 0.8]
            result_name = lambda value, seed: f"200_20_{value}_{seed}.pkl"  # noqa: E731
            fig_path = Path("figures/main") / f"tpr_{option}.pdf"
            xlabel = "signal"
            ylabel = "Power"
            is_null = False
            num_seeds = 1

    figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
    for value in values:
        results = Results()
        for seed in range(num_seeds):
            dir_path = Path(f"results_{option}")
            path = dir_path / result_name(value, seed)
            with path.open("rb") as f:
                results += pickle.load(f)
        assert len(results.p_values) == num_seeds * 1000

        figure.add_results(results.p_values, label="proposed", xloc=value)
        figure.add_results(results.oc_p_values, label="oc", xloc=value)
        if is_null:
            figure.add_results(results.naive_p_values, label="naive", xloc=value)

    if is_null:
        figure.add_red_line()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(fig_path, fontsize=16, legend_loc="upper left")


def plot_time(mode: str) -> None:
    """Plot the computation time of the experiments."""
    match mode:
        case "n":
            values = [100, 200, 300, 400]
            result_name = lambda value, seed: f"{value}_20_0.0_{seed}.pkl"  # noqa: E731
            fig_path = Path("figures/time") / "time_n.pdf"
            xlabel = "number of samples"
        case "d":
            values = [10, 20, 30, 40]
            result_name = lambda value, seed: f"200_{value}_0.0_{seed}.pkl"  # noqa: E731
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
            dir_path = Path("results_real")
            path = dir_path / f"{option}_{key}_{n}_{seed}.pkl"
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

            tpr = rejection_rate(results.p_values, alpha=0.05)
            oc_tpr = rejection_rate(results.oc_p_values, alpha=0.05)
            string += rf" & \textbf{{{tpr:.2f}}}/{oc_tpr:.2f}"
        strings_list.append(string)

    strings_path = Path(f"figures/real/{option}.txt")
    strings_path.parent.mkdir(parents=True, exist_ok=True)
    with strings_path.open("w") as f:
        f.write("\n".join(strings_list))


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
