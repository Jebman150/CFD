import numpy as np
import matplotlib.pyplot as plt


def plot_rows(
    filename,
    ylog: bool,
    title = "Missing title",
    xlabel = "",
    ylabel = ""
):
    plt.figure(figsize=(12, 6))

    with open(filename, "r") as file:
        for row_index, line in enumerate(file):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse values separated by whitespace
            y = np.array([float(value) for value in line.split()])

            # X values: 0, 1, 2, 3, ...
            x = np.arange(len(y))

            # Plot this row as a separate series
            plt.plot(x, y, marker=".", markersize=3,
                     linewidth=1.5, label=f"Row {row_index + 1}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ylog:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_rows(
        "logs/solver/solver_error.log",
        True,
        "Solver residual per iteration",
        "Iteration",
        "Residual")
    plot_rows(
        "logs/divergence_checker/max_divergence.log",
        True,
        "Max divergence in domain",
        "Frame",
        "max(div(v))")
    plot_rows(
        "logs/solver/iteration_count.log",
        False,
        "Iteration count",
        "Frame",
        "Solver iterations")