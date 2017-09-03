import numpy as np
from scipy.signal import lfilter

# http://www.johndcook.com/blog/standard_deviation/

class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self , shape , demean=True , destd=True , clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self , x , update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x , -self.clip , self.clip)
        return x

    def output_shape(self , input_space):
        return input_space.shape


class RunningStat(object):
    def __init__(self , shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self , x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


def test_running_stat():
    for shp in (() , (3 ,) , (3 , 4)):
        li = []
        rs = RunningStat(shp)
        for _ in range(5):
            val = np.random.randn(*shp)
            rs.push(val)
            li.append(val)
            m = np.mean(li , axis=0)
            assert np.allclose(rs.mean , m)
            v = np.square(m) if (len(li) == 1) else np.var(li , ddof=1 , axis=0)
            assert np.allclose(rs.var , v)

def explained_variance(ypred,y):
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
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


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
    return lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
