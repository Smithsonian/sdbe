#ifndef APHIDS_H
#define APHIDS_H

#include "hashpipe.h"
#include "hiredis/hiredis.h"

#define APHIDS_OK 0

#define APHIDS_CTX_INITIALIZED      1
#define APHIDS_CTX_NOT_INITIALIZED -1

#define APHIDS_ERR_INIT_FAIL  -1
#define APHIDS_ERR_CTX_UNINIT -2
#define APHIDS_UPDATE_EVERY 1000000

/* aphids_context_t

   This function initializes APHIDS functionality and should run 
   *inside* the init_method function of all APHIDS hashpipe threads.
*/
typedef struct aphids_context {
  int init;
  int iters;
  char *prefix;
  struct timeval begin, end;
  hashpipe_thread_args_t *thread_args;
  redisContext *redis_ctx;
} aphids_context_t;

/* aphids_init

   This function initializes APHIDS functionality and should either 
   be set *as* or run *inside* the init_method function of all APHIDS 
   hashpipe threads.
*/
int aphids_init(aphids_context_t * aphids_ctx, hashpipe_thread_args_t * thread_args);

/* aphids_update

   This function updates statistics related to the running thread such
   as number of iterations, run time, input data rate, etc. It should be
   run in the thread's loop on each iteration.
*/
int aphids_update(aphids_context_t * aphids_ctx);

/* aphids_destroy

   This function cleans up and destroys an APHIDS context and should
   at the end of all APHIDS-enable hashpipe threads.
*/
int aphids_destroy(aphids_context_t * aphids_ctx);
#include "aphids_db.h"

#endif
