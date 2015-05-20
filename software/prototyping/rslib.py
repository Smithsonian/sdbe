
from threading import Thread, Event
from Queue import Queue
from numpy.fft import rfft, irfft
from numpy import zeros, real, percentile, abs
import scipy
import sys
import os
import time
import vdif
import swarm
from argparse import Namespace
from sdbe_preprocess import get_diagnostics_from_file, process_chunk, run_diagnostics, \
                            quantize_to_2bit, vdif_psn_to_eud, vdif_station_id_str_to_int
import read_r2dbe_vdif
import struct

# in IF MHz (USB from 219122)
xs0_band = (7850, 9098)
xs1_band = (8150, 6902)
combined_band = (6976, 9024)

# I/O and prior processing
rel_path_dat = '/home/shared/sdbe_preprocessed/'
rel_path_out = '/home/shared/sdbe_preprocessed/'
scan_filename_base = 'prep6_test1_local'
# run_diagnostics(scan_filename_base, rel_path_dat, rel_path_out)
diagdict = get_diagnostics_from_file(scan_filename_base,rel_path_out)

stride = 1024 # units of snapshots
# Xs1 = phase sum 1 (channel 2 in xls spreadsheet), Xs0 = phase sum 0 (channel 1 in xls spreadsheet)
xs0_queue = Queue(maxsize=2*stride)
xs1_queue = Queue(maxsize=2*stride)
finished = Event()

# grab some number of Bframes off disk, put them as consecutive snapshots in snapshot queue
def snapshot_server():
    i = 0
    d = diagdict
    maxi = 2047
    while(True):
        xs1,xs0 = process_chunk(d['start1'],d['end1'],d['start2'],d['end2'], d['get_idx_offset'],
                                rel_path_dat + scan_filename_base,put_idx_range=[i,i+stride])
        for i in range(stride):
            xs0_queue.put(xs0[i,:])
            xs1_queue.put(xs1[i,:])
        i = i + stride
        if i > maxi:
            finished.set()
            break;

snapshot_thread = Thread(target=snapshot_server)
snapshot_thread.setDaemon(True)
snapshot_thread.start()

xs0_39snaps = Queue(maxsize=4)
xs1_39snaps = Queue(maxsize=4)

# take 39 snapshot spectra at a time, include nyq coeff, inverse transform to time domain, combine, and put back into freq domain
def snapshot_packager(inq, outq):
    while(True):
        if finished.isSet() and inq.qsize() < 39:
            break;
        timeseries = zeros((39,16384*2)) # last index is most rapidly varying
        for i in range(39):
            spectrum = zeros(16385, dtype='complex64')
            spectrum[0:16384] = inq.get()
            timeseries[i,:] = irfft(spectrum)
        outq.put(rfft(timeseries.ravel()))

snaps39_xs0_thread = Thread(target=snapshot_packager, args=(xs0_queue, xs0_39snaps))
snaps39_xs1_thread = Thread(target=snapshot_packager, args=(xs1_queue, xs1_39snaps))
snaps39_xs0_thread.setDaemon(True)
snaps39_xs1_thread.setDaemon(True)
snaps39_xs0_thread.start()
snaps39_xs1_thread.start()

fullband_out = Queue(maxsize=4)

# take the freq domain 39-snapshot spectra, join, and inverse transform to time domain at 4096 Msps
def merge_bands(inq0, inq1, outq):
    while(True):
        if finished.isSet() and inq0.empty() and inq1.empty(): # not really works
            break;
        xs0 = inq0.get() # phase sum 0 freq domain
        xs1 = inq1.get() # phase sum 1
        xslen = 39*16384 # freq 0 to nyq-1, freq spectra will be xslen+1 in length
        spectrum = zeros(2*xslen + 1 - 300*512, dtype='complex64') # 0-nyq USB, 1/512 MHz resolution, 6902-9098
        spectrum[0:(8000-6902)*512] = xs1[-1:150*512:-1] # reverse LSB spectrum, do not fill 8000 bin
        spectrum[(8000-6902)*512:] = xs0[150*512:] # fill 8000+ using USB of xs0 starting at 150
        # spectrum[(6976-6902)*512] = real(spectrum[(6976-6902)*512]) # make DC component real
        spectrum[(6976-6902)*512] = 0 # make DC component zero for unbiased signal, just in case
        spectrum[(9024-6902)*512] = real(spectrum[(9024-6902)*512]) # make nyq component real
        timeseries = irfft(spectrum[(6976-6902)*512:(9024-6902)*512+1])
        outq.put(timeseries)

fullband_thread = Thread(target=merge_bands, args=(xs0_39snaps, xs1_39snaps, fullband_out))
fullband_thread.setDaemon(True)
fullband_thread.start()

# quantize and write out as vdif
def write_vdif(inq, filename):
    out = open(filename, 'w')
    d = diagdict
    vdf_tmp = vdif.VDIFFrame.from_bin(struct.pack('<%dB' % read_r2dbe_vdif.FRAME_SIZE_BYTES,*d['vdif_template']))
    vdf_tmp.station_id = vdif_station_id_str_to_int('Sm')
    # tshift is added to t0 (ref epoch) to get acutal seconds of x[0] returned by data getting routine
    tshift = (d['get_idx_offset'][0] + d['start1'][0]) * (32768. / read_r2dbe_vdif.SWARM_RATE) + d['offset_swarmdbe_data'] * (1./4096e6)
    psn = int(round(tshift * 125000.)) # start with PSN corresponding to first data
    while(True):
        if finished.isSet() and fullband_out.empty():
            out.close()
            break;
        timeseries = inq.get() # 39 snapshots in 4096 Msps
        th = percentile(abs(timeseries), 68.2) # 1-sigma quantization threshold, assume zero mean
        tsquant = quantize_to_2bit(timeseries, th)
        subseries = tsquant.reshape((64, -1))
        for (j, series) in enumerate(subseries):
            df = psn % 125000 # number of 8us packets in 1s
            if(df == 0): # we looped over one second
                vdf_tmp.secs_since_epoch += 1
            vdf_tmp.eud[2:4] = vdif_psn_to_eud(vdf_tmp.psn)
            vdf_tmp.data_frame = df
            vdf_tmp.data = series
            out.write(vdf_tmp.to_bin())
            psn += 1

outfilename = "out.vdif"
vdif_thread = Thread(target=write_vdif, args=(fullband_out, outfilename))
vdif_thread.setDaemon(True)
vdif_thread.start()

def qinfo():
    qs = "xs0_queue xs1_queue xs0_39snaps xs1_39snaps fullband_out".split()
    for q in qs:
        qobj = globals()[q]
        print "%s: %d" % (q, qobj.qsize())
    threads = "snapshot_thread snaps39_xs0_thread snaps39_xs1_thread fullband_thread vdif_thread".split()
    for t in threads:
        tobj = globals()[t]
        print "%s: %s" % (t, repr(tobj.isAlive()))

