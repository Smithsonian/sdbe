#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "easy_databuf.h"

static void *run_method(hashpipe_thread_args_t * args)
{
	int rv = 0;
	easy_in_output_databuf_t *db_out = (easy_in_output_databuf_t *)args->obuf;
	hashpipe_status_t st = args->st;
	const char * status_key = args->thread_desc->skey;
	
	int idx_data = 0;
	char data = 'a';
	while (run_threads())
	{
		while ((rv=hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, idx_data)) != HASHPIPE_OK) {
			if (rv==HASHPIPE_TIMEOUT) {
				hashpipe_status_lock_safe(&st);
				hputs(st.buf, status_key, "blocked_in");
				hashpipe_status_unlock_safe(&st);
				continue;
			} else {
				hashpipe_error(__FUNCTION__, "error waiting for free databuf");
				pthread_exit(NULL);
				break;
			}
		}
		#ifdef DEBUG
			fprintf(stdout,"easy_in_thread:\n");
			fprintf(stdout,"\tcount = %d\n",db_out->count);
			fprintf(stdout,"\tdata[%d] = %c\n",idx_data,'a' + (char)(db_out->count % 26));
		#endif
		db_out->data[idx_data] = 'a' + (char)(db_out->count % 26);
		db_out->count++;
		hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, idx_data);
		idx_data = (idx_data + 1) % db_out->header.n_block;
		
		pthread_testcancel();
	}
	// Thread success!
	return NULL;
}

static hashpipe_thread_desc_t easy_in_thread = {
	name: "easy_in_thread",
	skey: "INSTAT",
	init: NULL,
	run:  run_method,
	ibuf_desc: {NULL},
	obuf_desc: {easy_buffer_create}
};

static __attribute__((constructor)) void ctor()
{
	register_hashpipe_thread(&easy_in_thread);
}
