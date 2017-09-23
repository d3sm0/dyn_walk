import numpy as np
from scipy.signal import lfilter


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def merge_dicts(x, y, z=None):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    if z:
        y = merge_dicts(y, z)
    z = x.copy()
    z.update(y)
    return z


import pickle
import glob

from tensorflow import logging


def train_test_set(datasets, split_size=.3):
    obs0 = datasets['obs'][:-1]
    obs1 = datasets['obs'][1:]
    acts = datasets['acts'][:-1]

    # TODO you may want to sample randomly here
    obs0_ts = obs0[:int(split_size * len(obs0))]
    obs1_ts = obs1[:int(split_size * len(obs1))]
    acts_ts = acts[:int(split_size * len(acts))]

    obs0_tr = obs0[int(split_size * len(obs0)):]
    obs1_tr = obs1[int(split_size * len(obs1)):]
    acts_tr = acts[int(split_size * len(acts)):]
    print("test length", int(split_size * len(obs0)), len(obs0))
    return (obs0_tr, acts_tr, obs1_tr), (obs0_ts, acts_ts, obs1_ts)


def load_data(data_dir):
    try:
        with open(data_dir + '/dataset.pkl', "rb") as fin:
            datasets = pickle.load(fin)
            logging.info('Dataset loadedd ')
    except IOError:
        logging.info('File not found, buidling dataset from default run')
        datasets = build_dataset(data_dir)
    return datasets


def build_dataset(data_dir):
    datasets = {}
    names = list(glob.glob(data_dir + "/dump_*"))
    i = 0
    for file_name in names:
        i += 1
        print(file_name, i, len(names), i / len(names))
        with open(file_name, "rb") as fin:
            data = pickle.load(fin, encoding="bytes")
            if len(datasets) == 0:
                for k, v in data.items():
                    datasets[k] = np.array(v)
            else:
                for k, v in data.items():
                    if np.ndim(v) == 0:
                        datasets[k] = np.append(datasets[k], v)
                    else:
                        datasets[k] = np.concatenate((datasets[k], v), axis=0)
    with open(data_dir + '/dataset.pkl', "wb") as fout:
        pickle.dump(datasets, fout)
        logging.info('Dataset saved')
    return datasets
