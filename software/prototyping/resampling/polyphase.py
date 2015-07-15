import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser
import numpy.fft as fft
import time
from fractions import gcd

# ipython -i polyphase.py -- -L 7 -M 11 -n 572 -f 150
# ipython -i polyphase.py -- -L 11 -M 7 -n 490

import sys
sdbe_scripts_dir = '/home/krosenfe/sdbe/software/prototyping'
sys.path.append(sdbe_scripts_dir)
import sdbe_preprocess
import read_r2dbe_vdif

def lcm(numbers):
    return reduce(lambda x, y: (x*y)/gcd(x,y), numbers, 1)

def pf_resample(b_poly,x,L,M):
  ''' Step through polyhase filters and decimate'''
  result = np.empty(x.size*L,dtype=x.dtype)
  for ind in range(L):
    result[ind::L] = signal.lfilter(b_poly[ind,:],[1],x)
  return result[::M]

  #result = np.empty(x.size*L/M)
  #for ind in range(L):
  #  result[ind::L] = signal.lfilter(b_poly[ind,:],[1],x)[???::M]
  #return result

if __name__ == "__main__":
  tic = time.time()

  # set up script
  parser = ArgumentParser(description='Resample SWARM data using a polyphase filter')
  parser.add_argument('-v',dest='verbose',action='store_true',help='display debug information')
  parser.add_argument('-b', dest='basename', help='scan filename base for SWARM data', default='prep6_test1_local')
  parser.add_argument('-d', dest='dir', help='directory for input SWARM data products', default='/home/shared/sdbe_preprocessed/')
  parser.add_argument('-n', type=int,dest='num_snapshots', help='number of snapshots to read', default=39*26)
  parser.add_argument('-s', type=int,dest='start_snapshot', help='start reading at snapshot index index', default=0)
  parser.add_argument('-L', type=int,dest='L', help='interpolation factor', default=32)
  parser.add_argument('-M', type=int,dest='M', help='decimation factor', default=39)
  parser.add_argument('-f', type=np.float,dest='freq_shift',help='frequency shift', default=150.)
  parser.add_argument('-np', type=int,dest='numtaps_poly',help='number of taps per sub-filter', default=31)
  parser.add_argument('-corr', dest='corr',help='correlate against R2DBE',action='store_true')
  args = parser.parse_args()

  # set up logger
  logger = logging.getLogger(__name__)
  logger.addHandler(logging.StreamHandler())
  logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

  assert (32768*args.num_snapshots*args.L) % args.M == 0, "Bad combination of resampling factor and number of samples"
  logger.info("Resampling factor (L/M): %d/%d" % (args.L,args.M))
  logger.info("Number of taps per sub-filter: %d" % args.numtaps_poly)
  logger.info("Modulating signal by %0.2f MHz" % args.freq_shift)

  # read B-frames and pre-computed diagnostics
  diagdict = sdbe_preprocess.get_diagnostics_from_file(args.basename,args.dir)
  d = diagdict
  Xs1,Xs0 = sdbe_preprocess.process_chunk(d['start1'],d['end1'],d['start2'],d['end2'], d['get_idx_offset'],
                        args.dir+args.basename,
                        put_idx_range=[args.start_snapshot,args.start_snapshot+args.num_snapshots])

  # read R2DBE data
  t_frame = read_r2dbe_vdif.R2DBE_SAMPLES_PER_WINDOW / read_r2dbe_vdif.R2DBE_RATE
  t_snapshot = read_r2dbe_vdif.SWARM_SAMPLES_PER_WINDOW / read_r2dbe_vdif.SWARM_RATE
  r2 = read_r2dbe_vdif.read_from_file(args.dir+args.basename+'_r2dbe_eth3.vdif',
                                        int(args.num_snapshots * t_snapshot / t_frame))

  # In anticipation of the SSB modulation, zero out the components that will get shifted past DC now.
  Xs1[:,:np.ceil(args.freq_shift/(read_r2dbe_vdif.SWARM_RATE / (Xs1.shape[-1]*2+1) / 1e6))] = 0

  # turn SWARM snapshots into timeseries data
  xs1 = fft.irfft(Xs1,n=32768,axis=-1)

  df = 1./t_snapshot/1e6 # frequency spacing in MHz
  freq1 = np.arange(0, 2496/2, df)

  # construct the polyphase filters 
  # the number of sub-filters = interpolation factor = args.L
  numtaps_total = args.L*args.numtaps_poly
  b = signal.firwin(numtaps_total,1./np.max([args.L,args.M]),window='hamming')
  b_poly = b.reshape((args.numtaps_poly,args.L)).T

  # SSB modulation
  # comput analytic signal using Hilbert transform.
  xs1_a = signal.hilbert(xs1.flatten())
  # modulate the signal
  xs1_a *= np.exp(-2j*np.pi*np.arange(xs1_a.size)/read_r2dbe_vdif.SWARM_RATE*args.freq_shift*1e6)
  # take the real part
  xs1_ssb = xs1_a.real

  # step through polyphase filter, including decimation step:
  xs1_resamp = pf_resample(b_poly,xs1_ssb,args.L,args.M)

  # inspect spectra
  num_resamp_samps_per_snap = 2**14
  _fold_n = xs1_resamp.size / num_resamp_samps_per_snap
  df_resamp = read_r2dbe_vdif.SWARM_RATE/1e6*args.L/args.M/num_resamp_samps_per_snap
  freq1_resamp= np.arange(0, read_r2dbe_vdif.SWARM_RATE/1e6/2*args.L/args.M+df_resamp, df_resamp)
  Xs1_resamp = fft.rfft(xs1_resamp[:_fold_n*num_resamp_samps_per_snap].reshape(_fold_n,num_resamp_samps_per_snap))

  toc = time.time()
  print 'took %0.2f seconds' % (toc-tic)

  # plot spectra
  plt.figure()
  plt.plot(freq1,(Xs1*Xs1.conj()).mean(0).real)
  plt.plot(freq1_resamp,Xs1.shape[-1]/(0.5*num_resamp_samps_per_snap)*args.M*args.L*(Xs1_resamp*Xs1_resamp.conj()).mean(0).real[:freq1_resamp.size])
  plt.xlim(0, 1248+200)
  plt.axvline(args.freq_shift)

  if args.corr:

    d = sdbe_preprocess.get_diagnostics_from_file(args.basename,rel_path=args.dir)
    offset_swarmdbe_data = d['offset_swarmdbe_data']
    idx_offset = d['get_idx_offset'][0]
    fft_window_size = 32768
    shift = np.floor((numtaps_total-1)/(2*args.L*read_r2dbe_vdif.SWARM_RATE)*2048e6)

    r2_x0 = sdbe_preprocess.bandlimit_1248_to_1024(r2[:(r2.size//4096)*4096],sub_sample=True)
    x1 = xs1_resamp[shift+offset_swarmdbe_data/2-1:]
    n = np.min((2**int(np.floor(np.log2(r2_x0.size))),2**int(np.floor(np.log2(x1.size)))))
    x0 = r2_x0[:n]
    x1 = x1[:n]
  
    X0 = fft.rfft(x0.reshape(n/fft_window_size,fft_window_size),axis=-1)
    X1 = fft.rfft(x1.reshape(n/fft_window_size,fft_window_size),axis=-1)
    X0*= np.exp(-1.0j*np.angle(X0[0,0]))
    X1*= np.exp(-1.0j*np.angle(X1[0,0]))
  
    S_0x0 = (X0 * X0.conj()).mean(axis=0).real
    S_1x1 = (X1 * X1.conj()).mean(axis=0).real
    S_0x1 = (X0 * X1.conj()).mean(axis=0).real
    p = 8
    s_0x0 = fft.irfft(S_0x0,n=p*fft_window_size).real
    s_1x1 = fft.irfft(S_1x1,n=p*fft_window_size).real
    s_0x1 = fft.irfft(S_0x1,n=p*fft_window_size).real/np.sqrt(s_0x0.max()*s_1x1.max())

    print ' (max,min) corr coeffs:\t\t', s_0x1.max(), s_0x1.min()

  #plt.ion()
  plt.show()
