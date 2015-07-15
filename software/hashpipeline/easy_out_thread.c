#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "easy_databuf.h"

static void *run_method(hashpipe_thread_args_t * args)
{
	int rv = 0;
	easy_in_output_databuf_t *db_in = (easy_in_output_databuf_t *)args->ibuf;
	hashpipe_status_t st = args->st;
	const char * status_key = args->thread_desc->skey;
	
	int idx_data = 0;
	while (run_threads())
	{
		while ((rv=hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, idx_data)) != HASHPIPE_OK) {
			if (rv==HASHPIPE_TIMEOUT) {
				hashpipe_status_lock_safe(&st);
				hputs(st.buf, status_key, "blocked_in");
				hashpipe_status_unlock_safe(&st);
				continue;
			} else {
				hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
				pthread_exit(NULL);
				break;
			}
		}
		if (db_in->count > 0)
		{
			#ifdef DEBUG
				fprintf(stdout,"easy_out_thread:\n");
				fprintf(stdout,"\tcount = %d\n",db_in->count);
				fprintf(stdout,"\tdata[%d] = %c\n",idx_data,db_in->data[idx_data]);
			#endif
			db_in->count--;
		}
		hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, idx_data);
		idx_data = (idx_data + 1) % db_in->header.n_block;
		fflush(stdout);
		
		pthread_testcancel();
	}
	
	// Thread success!
	return NULL;
}

static hashpipe_thread_desc_t easy_out_thread = {
	name: "easy_out_thread",
	skey: "INSTAT",
	init: NULL,
	run:  run_method,
	ibuf_desc: {easy_buffer_create},
	obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
	register_hashpipe_thread(&easy_out_thread);
}
