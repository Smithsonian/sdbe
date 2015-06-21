#ifndef APHIDS_C
#define APHIDS_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aphids.h"
#include "hiredis/hiredis.h"

/* aphids_init

   This function initializes APHIDS functionality and should run 
   *inside* the init_method function of all APHIDS hashpipe threads.
*/
int aphids_init(aphids_context_t * aphids_ctx, hashpipe_thread_args_t * thread_args) {

  char prefix[80];

  // set init flag to not initialized
  aphids_ctx->init = APHIDS_CTX_NOT_INITIALIZED;

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

  // we're done, initialize context
  aphids_ctx->init = APHIDS_CTX_INITIALIZED;

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

#endif
