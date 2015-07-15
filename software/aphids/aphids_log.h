#ifndef APHIDS_LOG_H
#define APHIDS_LOG_H

#include <stdarg.h>

#include "aphids.h"

#define APHIDS_LOG_OK    0
#define APHIDS_LOG_FAIL -1

#define APHIDS_LOG_DEBUG "debug"
#define APHIDS_LOG_INFO  "info"
#define APHIDS_LOG_ERROR "error"

/* aphids_log

   This function will publish a log entry using redis the pub/sub
   model. Note that in order to receive the log messages you will need 
   some kind of subscriber.
*/
int aphids_log(aphids_context_t * aphids_ctx, char * level, const char * fmt, ...);

#endif
