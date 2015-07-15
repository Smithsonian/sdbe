#ifndef EASY_DATABUF_H
#define EASY_DATABUF_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define EASY_IN_DATA_SIZE 16

typedef struct easy_in_output_databuf {
  hashpipe_databuf_t header;
  int count;
  char data[EASY_IN_DATA_SIZE];
} easy_in_output_databuf_t;

hashpipe_databuf_t *easy_buffer_create(int instance_id, int databuf_id);

#endif
