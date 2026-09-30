import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def draw_board(ax, perm, light="#f5d7b5", dark="#c9894a", queen="♛"):
    """Draw a board for the encoding ``perm[column] = row``."""
    n = len(perm)

    # Checkerboard
    board = np.indices((n, n)).sum(axis=0) % 2
    ax.imshow(board, cmap=ListedColormap([light, dark]), origin="upper")

    # X axis: columns 0..n-1
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(range(n))
    ax.xaxis.tick_top()

    # Y axis: rows 0..n-1
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(range(n))

    # Grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Queens: the gene position is the column and its value is the row.
    for column, row in enumerate(perm):
        ax.text(column, row, queen, ha="center", va="center", fontsize=28)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")