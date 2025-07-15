import numpy as np


def compute_distance(plotting_grids, normalize_to: (int, type(None)) = 1):
    """
    Plot the distance between two PDFs normalised to the standard deviation.
    """
    normalize_to -= 1  # Convert to 1-based index
    gr2_stats = plotting_grids[normalize_to]
    cv2 = gr2_stats.get_mean()
    sg2 = gr2_stats.get_std()
    N2 = gr2_stats.size

    distances = []
    for idx, grid in enumerate(plotting_grids):
        if idx == normalize_to:
            continue

        cv1 = grid.get_mean()
        sg1 = grid.get_std()
        N1 = grid.size

        # Wrap the distance into a Stats (1, flavours, points)
        distances.append(
            (grid.name, np.sqrt((cv1 - cv2) ** 2 / (sg1**2 / N1 + sg2**2 / N2)))
        )

    return distances


def gibbs_fn(x1, x2, delta, sigma, l0):
    """
    Gibbs kernel function for two points x1 and x2 with parameters delta, sigma, and l0.
    """

    def l(x):
        return l0 * (x + delta)

    return (
        x1
        * x2
        * sigma**2
        * np.sqrt(2 * l(x1) * l(x2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
        * np.exp(-np.power(x1 - x2, 2) / (np.power(l(x1), 2) + np.power(l(x2), 2)))
    )
