#ifndef BENG_REFORM_H
#define BENG_REFORM_H

#include <stdint.h>
#include "vdif_in_databuf.h"

// Define a B-engine timestamp type
typedef struct beng_timestamp {
	int32_t sec;
	int32_t clk;
} beng_timestamp_t;

// New B-engine header format
typedef struct newbeng_hdr {
  uint32_t b_upper;
  uint32_t c:8;
  uint32_t f:3;
  uint32_t b_lower:21;
} newbeng_hdr_t;

// Write the start-of-stream timestamp into t.
void get_beng_t0(beng_timestamp_t *t);

// Write the previous packet timestamp into t.
void get_beng_tprev(beng_timestamp_t *t);

// Write the previous packet B-frame counter value into b.
void get_beng_bprev(int64_t *b);

// Overwrite the B-engine headers of each vdif_in_packet_t instance so 
// that it conforms to the old-style, using the B-frame offset of the 
// B-engine timestamp in each packet compared to T0 as the B-count.
// Arguments:
// ----------
//   pkts -- pointer to B-engine-over-VDIF packets (unpacked 8pac data)
//   npkts -- number of packets stored in pkts
// Returns:
// --------
//   <void>
void beng_reform_headers(vdif_in_packet_t *pkts, int npkts);

#endif // BENG_REFORM_H
