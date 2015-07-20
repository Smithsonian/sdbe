#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"

hashpipe_databuf_t *vdif_in_databuf_create(int instance_id, int databuf_id)
{
  size_t header_size = sizeof(hashpipe_databuf_t);
  return hashpipe_databuf_create(instance_id, databuf_id, header_size, sizeof(vdif_in_packet_block_t), VDIF_IN_BUFFER_SIZE);
}

int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr) {
	int64_t b = 0;
	b |= ((int64_t)(vdif_pkt_hdr->beng.b_upper)&(int64_t)0x00000000FFFFFFFF) << 8;
	b |= (int64_t)(vdif_pkt_hdr->beng.b_lower)&(int64_t)0x00000000000000FF;
	return b;
}
