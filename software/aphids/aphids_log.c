#include <stdio.h>
#include <string.h>

#include "aphids_log.h"

int aphids_log(aphids_context_t * aphids_ctx, char * level, const char * fmt, ...) {

  redisContext *c = aphids_ctx->redis_ctx;
  redisReply *reply;
  char message[256];
  va_list args;

  // check that we're initialized
  if (aphids_ctx->init != APHIDS_CTX_INITIALIZED)
    return APHIDS_ERR_CTX_UNINIT;

  // format our log message
  va_start (args, fmt);
  vsprintf(message, fmt, args);
  va_end(args);

  // use the redis publish command
  reply = redisCommand(c, "PUBLISH aphids.%s %s", level, message);

  // check if the call failed
  if (!reply) {

    // we need to reconnect
    if (aphids_init(aphids_ctx, aphids_ctx->thread_args) != APHIDS_OK) {

      // can't reconnect, exit with error
      freeReplyObject(reply);
      return APHIDS_LOG_FAIL;

    }

  }

  // free reply and exit
  freeReplyObject(reply);
  return APHIDS_LOG_OK;
}
