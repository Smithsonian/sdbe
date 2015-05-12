from timeit import timeit

setup_cpu = """
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

setup_gpu = """
from numpy.fft import rfft, irfft
import numpy as np
n_swarm = 32768
n_swarm39 = n_swarm * 39
snapspersec = 2496e6 / n_swarm
n_r2dbe39 = 2097152 / 2
x_swarm = np.random.randn(n_swarm).astype(np.float32)
x_swarm39 = np.random.randn(n_swarm39)
f_swarm = rfft(x_swarm).astype(np.complex64)
t_swarm39 = np.arange(n_swarm39) / 2496.
t_r2dbe39 = np.arange(n_r2dbe39) / 2048.
inn = np.round(t_r2dbe39 * 2496.).astype(int)
ia = np.floor(t_r2dbe39 * 2496.).astype(int)
ib = np.ceil(t_r2dbe39 * 2496.).astype(int)
ib[(ia-ib == 0)] += 1
tt = t_swarm39[ib] - t_swarm39[ia]
cb = (t_r2dbe39 - t_swarm39[ia])/tt
ca = 1.-cb

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import scikits.cuda.fft as cu_fft

plan = cu_fft.Plan(f_swarm.size,np.complex64,np.float32)
d_f_swarm = gpuarray.to_gpu(f_swarm) 
d_x_swarm = gpuarray.empty(x_swarm.size,dtype=np.float32) 

plan39 = cu_fft.Plan(f_swarm.size,np.complex64,np.float32,batch=39)
d_f_swarm39 = gpuarray.to_gpu(np.tile(f_swarm,(39,1)) )
d_x_swarm39 = gpuarray.empty((39,x_swarm.size),dtype=np.float32) 
"""


exec(setup_cpu)

# inverse FFT one snap shot (complex to real)
test1 = """
y = irfft(f_swarm)
"""
n = 1000
t1 = timeit(test1, setup_cpu, number=n)
t1n = t1 * snapspersec / n

c_test1 = """
cu_fft.ifft(d_f_swarm,d_x_swarm,plan)
"""
n = 1000
c_t1 = timeit(c_test1, setup_gpu, number=n)
c_t1n = c_t1 * snapspersec / n

test2 = """
y = fftwirfft(f_swarm)
"""
n = 1000
t2 = timeit(test2, setup_cpu, number=n)
t2n = t2 * snapspersec / n

# inverse FFT 39 snap shots (complex to real)
c_test3 = """
cu_fft.ifft(d_f_swarm39,d_x_swarm39,plan39)
"""
n = 100
c_t3 = timeit(c_test3, setup_gpu, number=n)
c_t3n = c_t3 * snapspersec / 39. / n

# resample 39 snapshots
test3 = """
y = resample(x_swarm39, n_r2dbe39)
"""
n = 100
t3 = timeit(test3, setup_cpu, number=n)
t3n = t3 * snapspersec / 39. / n

test4 = """
y = x_swarm39[inn]
"""
n = 10000
t4 = timeit(test4, setup_cpu, number=n)
t4n = t4 * snapspersec / 39. / n

test5 = """
y = ca * x_swarm39[ia] + cb * x_swarm39[ib]
"""
n = 100
t5 = timeit(test5, setup_cpu, number=n)
t5n = t5 * snapspersec / 39. / n

