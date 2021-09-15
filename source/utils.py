import glob
import numpy as np

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def min_max_norm(x):

    return (x - x.min() + (1e-30)) / (x.max() - x.min() + (1e-30))

def bayesian_sampling(theta_bank):

    [swa, diag, dcol] = theta_bank

    z1 = np.random.normal(size=diag.shape[0])
    z2 = np.random.normal(size=len(dcol))
    term1 = swa
    term2 = (np.sqrt(np.maximum(diag, 1e-30)) * z1) / np.sqrt(2)
    term3 = np.matmul(np.asarray(dcol).T, z2) / np.sqrt(2 * (len(dcol) - 1))
    sample = term1 + term2 + term3

    return sample
