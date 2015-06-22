'''
Compare CUDA resampling to python results
'''

import sys
sdbe_scripts_dir = '/home/krosenfe/sdbe/software/prototyping'
sys.path.append(sdbe_scripts_dir)

import read_sdbe_vdif, read_r2dbe_vdif, cross_corr
import numpy as np
import logging

from sdbe_preprocess import get_diagnostics_from_file,process_chunk,resample_sdbe_to_r2dbe_fft_interp,run_diagnostics

if __name__ == "__main__":
	scan_filename_base = 'prep6_test1_local'
	rel_path_dat = '/home/shared/sdbe_preprocessed/' 

        # turn on logging
        logfilename = scan_filename_base + '_ppsdbe.log'
        #logging.basicConfig(format=logging.BASIC_FORMAT,level=logging.INFO,stream=sys.stdout)
        logging.basicConfig(format=logging.BASIC_FORMAT,level=logging.INFO,filename=logfilename)
        logger = logging.getLogger()

	fh = run_diagnostics(scan_filename_base,rel_path_to_in=rel_path_dat,rel_path_to_out='./')

	# read diagnostics output
	d = get_diagnostics_from_file(scan_filename_base,rel_path=rel_path_dat)

	# processing is done multiples of 39-snapshot frames so the time-domain 
	# signal given to bandlimit_1248_to_1024 is a power-of-two element array.
	chunks_in_39_snapshots = 128
	t_chunk = chunks_in_39_snapshots*39*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW/read_sdbe_vdif.SWARM_RATE

	ichunk = 0
	chunk_start = ichunk * chunks_in_39_snapshots * 39
	chunk_stop = (ichunk+1) * chunks_in_39_snapshots * 39
	Xs1,Xs0 = process_chunk(d['start1'],d['end1'],d['start2'],d['end2'],d['get_idx_offset'],
			rel_path_dat + scan_filename_base,put_idx_range=[chunk_start,chunk_stop])

	# resample
	xs1 = resample_sdbe_to_r2dbe_fft_interp(Xs1,interp_kind="linear")

	plt.ion()
