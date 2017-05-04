#ifndef SAMPLE_RATES_H
#define SAMPLE_RATES_H

// these just set some constants for identification
#define SWARM_RATE_FRAC_6_11 6
#define SWARM_RATE_FRAC_8_11 8
#define SWARM_RATE_FRAC_10_11 10
#define SWARM_RATE_FRAC_11_11 11
#define SWARM_RATE_SPINAL_TAP 11

// choose correct sample rate here
#ifndef SWARM_RATE_FRAC
#define SWARM_RATE_FRAC SWARM_RATE_FRAC_11_11
#endif

// rates
#define SWARM_RATE_FULL 4576e6
#define R2DBE_RATE 4096e6

#define SWARM_RATE (SWARM_RATE_FULL*SWARM_RATE_FRAC/SWARM_RATE_SPINAL_TAP)
#if SWARM_RATE_FRAC == SWARM_RATE_FRAC_6_11
	// resampling factors
	#define DECIMATION_FACTOR 39
	#define EXPANSION_FACTOR 32
	// time alignment
	#define MAGIC_OFFSET_IN_BENG_FFT_WINDOWS (52) // see vdif_in_databuf.c for details
	// number of VDIF frames produced in output stream for one block of
	// B-engine frame at input (see vdif_out_databuf.h for more info)
	#define VDIF_OUT_PKTS_PER_BLOCK 8192
#elif SWARM_RATE_FRAC == SWARM_RATE_FRAC_8_11
	// resampling factors
	#define DECIMATION_FACTOR 13
	#define EXPANSION_FACTOR 8
	// time alignment, value confirmed to be the same as for 6/11 using two different datasets
	#define MAGIC_OFFSET_IN_BENG_FFT_WINDOWS (52) // see vdif_in_databuf.c for details
	// number of VDIF frames produced in output stream for one block of
	// B-engine frame at input (see vdif_out_databuf.h for more info)
	#define VDIF_OUT_PKTS_PER_BLOCK 2048
#elif SWARM_RATE_FRAC == SWARM_RATE_FRAC_10_11
	// resampling factors
	#define DECIMATION_FACTOR 65
	#define EXPANSION_FACTOR 64
	// time alignment, value still unconfirmed!!!!!
	#define MAGIC_OFFSET_IN_BENG_FFT_WINDOWS (52) // see vdif_in_databuf.c for details
	// number of VDIF frames produced in output stream for one block of
	// B-engine frame at input (see vdif_out_databuf.h for more info)
	#define VDIF_OUT_PKTS_PER_BLOCK 8192
#elif SWARM_RATE_FRAC == SWARM_RATE_FRAC_11_11
	// THIS SET OF PARAMETERS WILL LIKELY OVERFLOW THE MEMORY ON THE GPU UNTIL SOME 
	// CHANGES ARE MADE TO THE BATCH SIZES / BUFFER ARCHITECTURE.
	// resampling factors
	#define DECIMATION_FACTOR 143
	#define EXPANSION_FACTOR 128
	// time alignment, value still unconfirmed!!!!!
	#define MAGIC_OFFSET_IN_BENG_FFT_WINDOWS (52) // see vdif_in_databuf.c for details
	// number of VDIF frames produced in output stream for one block of
	// B-engine frame at input (see vdif_out_databuf.h for more info)
	// (DECIMATION_FACTOR*BENG_SNAPSHOTS*(2*BENG_CHANNELS_) * (EXPANSION_FACTOR/DECIMATION_FACTOR) / <samples-per-packet-out>
	#define VDIF_OUT_PKTS_PER_BLOCK 16384
#else
	// invalid SWARM rate, error
	#error "Invalid SWARM rate, aborting compilation"
#endif

/* This defines how many B-frames are needed to fill a complete 
 * DECIMATION_FACTOR. Since the data reordering can be described as a 
 * circular shift of data within a single B-frame, the number of frames 
 * is now exactly equal to the value of DECIMATION_FACTOR.
 */
#define BENG_FRAMES_PER_GROUP DECIMATION_FACTOR

// Number of VDIF frames per second in output stream (not yet dependent
// on the SWARM x/11 rate)
#define VDIF_OUT_FRAMES_PER_SECOND 125000

#endif // SAMPLE_RATES_H

