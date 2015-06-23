#ifndef APHIDS_H
#define APHIDS_H

#include "hashpipe.h"
#include "hiredis/hiredis.h"

#define APHIDS_OK 0

#define APHIDS_CTX_INITIALIZED      1
#define APHIDS_CTX_NOT_INITIALIZED -1

#define APHIDS_ERR_INIT_FAIL  -1
#define APHIDS_ERR_CTX_UNINIT -2
#define APHIDS_ERR_GET_FAIL   -3
#define APHIDS_ERR_SET_FAIL   -4

#define APHIDS_REDIS_HOST "127.0.0.1"
#define APHIDS_REDIS_PORT 6379
#define APHIDS_REDIS_TIMEOUT 1 // seconds

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

/* aphids_set, aphids_get

   These function set an APHIDS key to a certain value on a local redis 
   server. The keys have a prefix assigned to them that depends on the thread
   name and hashpipe instance ID, the keys follow this pattern:

   aphids[<instance_id>]:<thread_name>:<key>

*/
int aphids_set(aphids_context_t * aphids_ctx, char * key, char * value);
int aphids_get(aphids_context_t * aphids_ctx, char * key, char * value);

/* aphids_destroy

   This function cleans up and destroys an APHIDS context and should
   at the end of all APHIDS-enable hashpipe threads.
*/
int aphids_destroy(aphids_context_t * aphids_ctx);

#endif
