#ifndef APHIDS_H
#define APHIDS_H

#include <stdarg.h>

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

/* aphids_context_t

   This function initializes APHIDS functionality and should run 
   *inside* the init_method function of all APHIDS hashpipe threads.
*/
typedef struct aphids_context {
  int init;
  char *prefix;
  hashpipe_thread_args_t *thread_args;
  redisContext *redis_ctx;
} aphids_context_t;

/* aphids_init

   This function initializes APHIDS functionality and should either 
   be set *as* or run *inside* the init_method function of all APHIDS 
   hashpipe threads.
*/
int aphids_init(aphids_context_t * aphids_ctx, hashpipe_thread_args_t * thread_args);

/* aphids set

   This function sets an APHIDS key to a certain value.
*/
int aphids_set(aphids_context_t * aphids_ctx, char * key, char * value);
int aphids_get(aphids_context_t * aphids_ctx, char * key, char * value);

#endif
