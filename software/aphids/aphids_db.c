#include <string.h>

#include "aphids_db.h"

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

int aphids_put(aphids_context_t * aphids_ctx, char * key, char * value) {

  redisReply *reply;
  redisContext *c = aphids_ctx->redis_ctx;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // use the redis right push command
  reply = redisCommand(c, "RPUSH %s:%s %s", aphids_ctx->prefix, key, value);

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

int aphids_pop(aphids_context_t * aphids_ctx, char * key, char * value, int timeout) {

  redisReply *reply;
  redisContext *c = aphids_ctx->redis_ctx;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // use the redis block left pop command
  reply = redisCommand(c, "BLPOP %s:%s %d", aphids_ctx->prefix, key, timeout);

  // check if the call failed
  if (!reply) {

    // we need to reconnect
    if (aphids_init(aphids_ctx, aphids_ctx->thread_args) != APHIDS_OK) {

      // can't reconnect, exit with error
      freeReplyObject(reply);
      return APHIDS_ERR_POP_FAIL;

    }

  }

  // check if the call timed out
  if (reply->type == REDIS_REPLY_NIL) {
    return APHIDS_POP_TIMEOUT;
  }

  // check the reply type
  if (reply->type != REDIS_REPLY_ARRAY) {

    // something is wrong, exit with error
    freeReplyObject(reply);
    return APHIDS_ERR_POP_FAIL;

  }

  // finally copy over the reply
  strcpy(value, reply->element[1]->str);

  // free reply and exit
  freeReplyObject(reply);
  return APHIDS_OK;
}
