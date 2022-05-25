import matplotlib.pyplot as plt
from typing import List


def default_plot(dfs: List, variable: str, labels: List = None) -> None:
    """Basic plot function that takes a number of DataFrames and plots the 'variable' columns in one figure"""
    plt.figure(figsize=(12, 4))
    for ix, df in enumerate(dfs):
        if labels is None:
            plt.plot(df[variable].dropna())
        else:
            plt.plot(df[variable].dropna(), label=labels[ix])

    if labels is not None:
        plt.legend()

    plt.gca().update(
        dict(title=r"Plot of: " + variable, xlabel="date", ylabel=variable)
    )
    plt.grid()
    plt.show()
