import numpy as np
import matplotlib.pyplot as plt


def plot_rows(filename):
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

    plt.xlabel("X (value index)")
    plt.ylabel("Value")
    plt.title("Values per Row")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_rows("log/solver/solver_error.log")