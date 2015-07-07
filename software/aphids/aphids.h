#ifndef APHIDS_H
#define APHIDS_H

#include "hashpipe.h"
#include "hiredis/hiredis.h"

#define APHIDS_OK 0

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

#include "aphids_db.h"
#include "aphids_log.h"
#include "aphids_loop.h"

#endif
