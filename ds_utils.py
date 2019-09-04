from sklearn.metrics.pairwise import linear_kernel
import numpy as np


# To deal with large memory data, chunk method is necessary
def chunk_linear_kernel(matrix, chunk_size):
    res = []
    max_len = matrix.shape[0]
    for start in range(0, max_len, chunk_size):
        end = min(max_len, start+chunk_size)
        cos_sim = linear_kernel(matrix[start:end], matrix)
        res.append(cos_sim)

    return np.vstack(res)

