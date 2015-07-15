#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_out_databuf.h"

hashpipe_databuf_t *vdif_out_databuf_create(int instance_id, int databuf_id)
{
  size_t header_size = sizeof(hashpipe_databuf_t);
  return hashpipe_databuf_create(instance_id, databuf_id, header_size, sizeof(vdif_out_packet_block_t), VDIF_OUT_BUFFER_SIZE);
}
