"""Monotonic DTW, from https://github.com/dgaddy/silent_speech."""
import numpy as np
from numba import jit


@jit(nopython=True)
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1]):
            dtw[i, j] = costs[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw


def align_from_distances(distance_matrix):
    """For every row, the column it is aligned to on the best monotonic path."""
    dtw = time_warp(distance_matrix)
    i = distance_matrix.shape[0] - 1
    j = distance_matrix.shape[1] - 1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i - 1, j), (i, j - 1), (i - 1, j - 1)], key=lambda x: dtw[x[0], x[1]])
    return results
