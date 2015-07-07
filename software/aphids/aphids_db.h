#ifndef APHIDS_DB_H
#define APHIDS_DB_H

#include "aphids.h"

#define APHIDS_ERR_GET_FAIL   -3
#define APHIDS_ERR_SET_FAIL   -4

#define APHIDS_REDIS_HOST "192.168.10.10"
#define APHIDS_REDIS_PORT 6379
#define APHIDS_REDIS_TIMEOUT 1 // seconds

/* aphids_set, aphids_get

   These function set an APHIDS key to a certain value on a local redis
   server. The keys have a prefix assigned to them that depends on the thread
   name and hashpipe instance ID, the keys follow this pattern:

   aphids[<instance_id>]:<thread_name>:<key>

*/
int aphids_set(aphids_context_t * aphids_ctx, char * key, char * value);
int aphids_get(aphids_context_t * aphids_ctx, char * key, char * value);

#endif

