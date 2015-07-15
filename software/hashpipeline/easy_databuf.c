#ifndef EASY_DATABUF_C
#define EASY_DATABUF_C

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "easy_databuf.h"

hashpipe_databuf_t *easy_buffer_create(int instance_id, int databuf_id)
{
	size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(int);
	return hashpipe_databuf_create(instance_id, databuf_id, header_size, sizeof(char), EASY_IN_DATA_SIZE);
}


#endif
