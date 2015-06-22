#ifndef APHIDS_C
#define APHIDS_C

#include <stdio.h>
#include <syslog.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "aphids.h"
#include "hiredis/hiredis.h"


int aphids_init(aphids_context_t * aphids_ctx, hashpipe_thread_args_t * thread_args) {

  char prefix[80];
  hashpipe_status_t st = thread_args->st;
  const char * status_key = thread_args->thread_desc->skey;

  // set init flag to not initialized
  aphids_ctx->init = APHIDS_CTX_NOT_INITIALIZED;

  // initialize iters
  aphids_ctx->iters = 0;

  // set the redis key prefix string
  sprintf(prefix, "aphids[%d]:%s", thread_args->instance_id, thread_args->thread_desc->name);
  aphids_ctx->prefix = malloc(strlen(prefix));
  strcpy(aphids_ctx->prefix, prefix);

  // add the thread arguments to context
  aphids_ctx->thread_args = thread_args;

  // set timeout interval
  struct timeval timeout = {APHIDS_REDIS_TIMEOUT, 0};

  // connect to redis server with timeout
  redisContext *c = redisConnectWithTimeout(APHIDS_REDIS_HOST, APHIDS_REDIS_PORT, timeout);
  if (c == NULL || c->err)
    return APHIDS_ERR_INIT_FAIL;

  // add redis context to aphids context
  aphids_ctx->redis_ctx = c;

  // open syslog so we can send messages
  openlog("aphids", LOG_PID, LOG_USER);

  // log in syslog, just to say we're starting
  syslog(LOG_INFO, "%s started", aphids_ctx->prefix);

  // update our hashpipe status
  hashpipe_status_lock_safe(&st);
  hputs(st.buf, status_key, "running");
  hashpipe_status_unlock_safe(&st);

  // we're done, initialize context
  aphids_ctx->init = APHIDS_CTX_INITIALIZED;

  return APHIDS_OK;
}

int aphids_update(aphids_context_t * aphids_ctx) {

  char aphids_val[80];
  double seg_time, seg_rate=0.0;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // update begin time on the first time
  if (aphids_ctx->iters == 0) {
    gettimeofday(&aphids_ctx->begin, NULL);
  }

  // increment iter counter
  aphids_ctx->iters++;

  if ((aphids_ctx->iters % APHIDS_UPDATE_EVERY) == 0) {
    // only do this every so often

    gettimeofday(&aphids_ctx->end, NULL); // take note of current time

    // and calculate time spent in the last time segment
    seg_time = (double)(aphids_ctx->end.tv_usec - aphids_ctx->begin.tv_usec) / 1e6 +
      (double)(aphids_ctx->end.tv_sec - aphids_ctx->begin.tv_sec);

    // if thread has input buffer
    if (aphids_ctx->thread_args->ibuf) {

      // calculate total buffer bit-rate (in Gbps)
      seg_rate = APHIDS_UPDATE_EVERY * 1e-9 *
	(double)(aphids_ctx->thread_args->ibuf->block_size * 8) / seg_time;

      // set rate over last segment
      sprintf(aphids_val, "%f Gbps", seg_rate);
      aphids_set(aphids_ctx, "ibuf:seg_rate", aphids_val);

    }

    // if thread has output buffer
    if (aphids_ctx->thread_args->obuf) {

      // calculate total buffer bit-rate (in Gbps)
      seg_rate = APHIDS_UPDATE_EVERY * 1e-9 *
	(double)(aphids_ctx->thread_args->obuf->block_size * 8) / seg_time;

      // set rate over last segment
      sprintf(aphids_val, "%f Gbps", seg_rate);
      aphids_set(aphids_ctx, "obuf:seg_rate", aphids_val);

    }

    // set number of iterations
    sprintf(aphids_val, "%d", aphids_ctx->iters);
    aphids_set(aphids_ctx, "iters", aphids_val);


    // set time over last segment
    sprintf(aphids_val, "%f secs", seg_time);
    aphids_set(aphids_ctx, "seg_time", aphids_val);

    aphids_ctx->begin = aphids_ctx->end; // keep the time for the next update

  } // end if

  return APHIDS_OK;
}

int aphids_set(aphids_context_t * aphids_ctx, char * key, char * value) {

  redisReply *reply;
  redisContext *c = aphids_ctx->redis_ctx;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // use the redis set command
  reply = redisCommand(c, "SET %s:%s %s", aphids_ctx->prefix, key, value);

  // check if the call failed
  if (!reply) {

    // we need to reconnect
    if (aphids_init(aphids_ctx, aphids_ctx->thread_args) != APHIDS_OK) {

      // can't reconnect, exit with error
      freeReplyObject(reply);
      return APHIDS_ERR_SET_FAIL;

    }

  }

  // free reply and exit
  freeReplyObject(reply);
  return APHIDS_OK;
}  

int aphids_get(aphids_context_t * aphids_ctx, char * key, char * value) {

  redisReply *reply;
  redisContext *c = aphids_ctx->redis_ctx;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // use the redis set command
  reply = redisCommand(c, "GET %s:%s", aphids_ctx->prefix, key);

  // check if the call failed
  if (!reply) {

    // we need to reconnect
    if (aphids_init(aphids_ctx, aphids_ctx->thread_args) != APHIDS_OK) {

      // can't reconnect, exit with error
      freeReplyObject(reply);
      return APHIDS_ERR_GET_FAIL;

    }

  }

  // check the reply type
  if (reply->type != REDIS_REPLY_STRING) {

    // something is wrong, exit with error
    freeReplyObject(reply);
    return APHIDS_ERR_GET_FAIL;

  }

  // finally copy over the reply
  strcpy(value, reply->str);

  // free reply and exit
  freeReplyObject(reply);
  return APHIDS_OK;
}  

int aphids_destroy(aphids_context_t * aphids_ctx) {

  hashpipe_status_t st = aphids_ctx->thread_args->st;
  const char * status_key = aphids_ctx->thread_args->thread_desc->skey;

  // update our status
  hashpipe_status_lock_safe(&st);
  hputs(st.buf, status_key, "stopping");
  hashpipe_status_unlock_safe(&st);

  // free redis context
  redisFree(aphids_ctx->redis_ctx);

  // one last log, number iterations
  syslog(LOG_INFO, "%s stopped after %d iterations",
	 aphids_ctx->thread_args->thread_desc->name, aphids_ctx->iters);

  closelog(); // close logger

  return APHIDS_OK;

}

#endif
