#include "beng_reform.h"
#include "vdif_in_databuf.h"

// This is the timestamp associated with the first valid B-engine packet 
// received.
static beng_timestamp_t T0 = {-1,-1};

// This is the timestamp associated with the last valid B-engine packet
// received.
static beng_timestamp_t TPREV = {-1,-1};

// This is the B-engine frame counter relative to 0 at the first valid 
// B-engine packet received.
static int64_t BPREV = -1;

void _read_beng_timestamp(const newbeng_hdr_t *hdr, beng_timestamp_t *t) {
	int64_t b = 0;
	b |= (int64_t)(hdr->b_upper) << 21;
	b |= (int64_t)(hdr->b_lower & 0x001FFFFF);
	t->clk = (int32_t)( b & (int64_t)0x000000001FFFFFFF);
	t->sec = (int32_t)((b & (int64_t)0x001FFFFFE0000000)>>29);
}

// Number of FPGA clock cycles per second in SWARM
#define SWARM_CLK_PER_SEC 286000000
// Number of FPGA clock cycles per B-engine frame
#define SWARM_BFRAME_CLK 262144
// FPGA clock count difference less than this will be considered negligible
#define BFRAME_CLK_TOL 1024

// Compare timestamps and return a value less than, equal to, or 
// greater than zero corresponding to whether then is considered to be 
// earlier than, coinciding with, or later than now.
int _beng_timestamp_compare_then_now(const beng_timestamp_t *then, const beng_timestamp_t *now) {
	if (then->sec < now->sec) {
		if (then->sec == now->sec-1) {
			// check if within a clock offset tolerance
			if (then->clk + BFRAME_CLK_TOL > SWARM_CLK_PER_SEC && (then->clk + BFRAME_CLK_TOL)%SWARM_CLK_PER_SEC >= now->clk) {
				return 0;
			}
		}
		// if this point reached, then is definitely earlier
		return -1;
	} else if (then->sec > now->sec) {
			// check if within a clock offset tolerance
			if (now->clk + BFRAME_CLK_TOL > SWARM_CLK_PER_SEC && (now->clk + BFRAME_CLK_TOL)%SWARM_CLK_PER_SEC >= then->clk) {
				return 0;
			}
		// if this point reached, then is definitely later
		return 1;
	} else {
		// sec values are equal, just compare the clk values
		if (then->clk + BFRAME_CLK_TOL < now->clk) {
			return -1;
		} else if (now->clk + BFRAME_CLK_TOL < then->clk) {
			return 1;
		} else {
			return 0;
		}
	}
}

// Maximum B-frame offset that will actually be counted
#ifndef MAX_COUNTABLE_BFRAME_OFFSET
#define MAX_COUNTABLE_BFRAME_OFFSET 1000
#endif
// Return the number of B-engine frame offset of now relative to then. 
// The returned value is signed, and zero meaning the timestamps are 
// equivalent. The maximum offset is limited to the value of the 
// preprocessor variable MAX_COUNTABLE_BFRAME_OFFSET.
int64_t _beng_timestamp_offset_then_now(beng_timestamp_t *then, beng_timestamp_t *now) {
	int64_t offset = 0;
	int step = 0;
	int comparison;
	beng_timestamp_t t_start, t_end;
	comparison = _beng_timestamp_compare_then_now(then,now);
	if (comparison < 0) {
		// then earlier than now. set start to then and end to now
		t_start.sec = then->sec;
		t_start.clk = then->clk;
		t_end.sec = now->sec;
		t_end.clk = now->clk;
		step = 1;
	} else if (comparison > 0) {
		// then later than now. set start to now and end to then
		t_start.sec = now->sec;
		t_start.clk = now->clk;
		t_end.sec = then->sec;
		t_end.clk = then->clk;
		step = -1;
	} else {
		// timestamps equal, zero offset
		return (int64_t)0;
	}
	// increment start by SWARM_BFRAME_CLK until it is equivalent to end
	while (_beng_timestamp_compare_then_now(&t_start,&t_end) < 0) {
		beng_timestamp_increment(&t_start);
		offset += (int64_t)step;
		if (offset > MAX_COUNTABLE_BFRAME_OFFSET || -offset > MAX_COUNTABLE_BFRAME_OFFSET) {
			break;
		}
	}
	return offset;
}

/* This constant determines by how many B-engine frames the timestamp in
 * B-engine data is off from real-time. A positive value means the
 * corrected timestamp is achived with _beng_timestamp_increment;
 * conversely, a negative value means the corrected timestamp is
 * achieved with _beng_timestamp_decrement.
 */
#define SWARM_TIMESTAMP_OFF_BY_BENG -1
void get_beng_t0(beng_timestamp_t *t) {
	int ii;
	t->sec = T0.sec;
	t->clk = T0.clk;
	if (SWARM_TIMESTAMP_OFF_BY_BENG < 0) {
		for (ii=0; ii>SWARM_TIMESTAMP_OFF_BY_BENG; ii--) {
			beng_timestamp_decrement(t);
		}
	} else if (SWARM_TIMESTAMP_OFF_BY_BENG > 0) {
		for (ii=0; ii<SWARM_TIMESTAMP_OFF_BY_BENG; ii++) {
			beng_timestamp_increment(t);
		}
	}
}

void get_beng_tprev(beng_timestamp_t *t) {
	t->sec = TPREV.sec;
	t->clk = TPREV.clk;
}

void get_beng_bprev(int64_t *b) {
	*b = BPREV;
}

void beng_reform_headers(vdif_in_packet_t *pkts, int npkts) {
	int ii;
	int bframe_offset;
	int f,c;
	newbeng_hdr_t *hdr;
	beng_timestamp_t t_now;
	vdif_in_packet_t *pkt;
	if (T0.sec == -1 && T0.clk == -1) {
		hdr = (newbeng_hdr_t *)&(pkts->header.beng);
		_read_beng_timestamp(hdr,&T0);
		TPREV.sec = T0.sec;
		TPREV.clk = T0.clk;
		BPREV = 0;
	}
	for (ii=0; ii<npkts; ii++) {
		// point to next packet
		pkt = pkts+ii;
		// ignore invalid packets
		if (pkt->header.w0.invalid) {
			continue;
		}
		// get pointer to header, as new format
		hdr = (newbeng_hdr_t *)&(pkt->header.beng);
		// read the timestamp
		_read_beng_timestamp(hdr,&t_now);
		// count B-frame offset since previous packet...
		bframe_offset = _beng_timestamp_offset_then_now(&TPREV,&t_now);
		if (bframe_offset > MAX_COUNTABLE_BFRAME_OFFSET || bframe_offset < -MAX_COUNTABLE_BFRAME_OFFSET) {
			// ...if beyond maximum countable, just mark packet invalid
			pkt->header.w0.invalid = 1;
			bframe_offset = -1;
		} else {
			// ...and add it to B-frame offset of previous packet
			bframe_offset += BPREV;
			// update the previous timestamp and B-frame offset values
			TPREV.sec = t_now.sec;
			TPREV.clk = t_now.clk;
			BPREV = bframe_offset;
		}
		// read FID, CID
		c = hdr->c;
		f = hdr->f;
		// write header information according to old format
		pkt->header.beng.f = f;
		pkt->header.beng.c = c;
		pkt->header.beng.b_lower =  bframe_offset & 0x000000FF;
		pkt->header.beng.b_upper = (bframe_offset & 0x7FFFFF00)>>8;
	}
}

double beng_timestamp_clk_to_float(beng_timestamp_t *t) {
	return (double)(t->clk) / (double)SWARM_CLK_PER_SEC;
}

void beng_timestamp_increment(beng_timestamp_t *t) {
	t->clk += SWARM_BFRAME_CLK;
	if (t->clk > SWARM_CLK_PER_SEC) {
		t->sec += 1;
		t->clk = t->clk % SWARM_CLK_PER_SEC;
	}
}

void beng_timestamp_decrement(beng_timestamp_t *t) {
	t->clk -= SWARM_BFRAME_CLK;
	if (t->clk < 0) {
		t->sec -= 1;
		t->clk = t->clk + SWARM_CLK_PER_SEC;
	}
}
