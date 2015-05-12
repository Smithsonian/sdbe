from timeit import timeit

setup1 = """
from scipy.signal import resample
from pyfftw.interfaces import numpy_fft as fftw
fftwrfft = fftw.rfft
fftwirfft = fftw.irfft
from numpy.fft import rfft, irfft
import numpy as np
n_swarm = 32768
n_swarm39 = n_swarm * 39
snapspersec = 2496e6 / n_swarm
n_r2dbe39 = 2097152 / 2
x_swarm = np.random.randn(n_swarm)
x_swarm39 = np.random.randn(n_swarm39)
f_swarm = rfft(x_swarm)
t_swarm39 = np.arange(n_swarm39) / 2496.
t_r2dbe39 = np.arange(n_r2dbe39) / 2048.
inn = np.round(t_r2dbe39 * 2496.).astype(int)
ia = np.floor(t_r2dbe39 * 2496.).astype(int)
ib = np.ceil(t_r2dbe39 * 2496.).astype(int)
ib[(ia-ib == 0)] += 1
tt = t_swarm39[ib] - t_swarm39[ia]
cb = (t_r2dbe39 - t_swarm39[ia])/tt
ca = 1.-cb
"""

exec(setup1)

test1 = """
y = irfft(f_swarm)
"""
n = 1000
t1 = timeit(test1, setup1, number=n)
t1n = t1 * snapspersec / n

test2 = """
y = fftwirfft(f_swarm)
"""
n = 1000
t2 = timeit(test2, setup1, number=n)
t2n = t2 * snapspersec / n

test3 = """
y = resample(x_swarm39, n_r2dbe39)
"""
n = 100
t3 = timeit(test3, setup1, number=n)
t3n = t3 * snapspersec / 39. / n

test4 = """
y = x_swarm39[inn]
"""
n = 10000
t4 = timeit(test4, setup1, number=n)
t4n = t4 * snapspersec / 39. / n

test5 = """
y = ca * x_swarm39[ia] + cb * x_swarm39[ib]
"""
n = 100
t5 = timeit(test5, setup1, number=n)
t5n = t5 * snapspersec / 39. / n

