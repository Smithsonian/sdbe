#ifndef APHIDS_DB_H
#define APHIDS_DB_H

#include "aphids.h"

#define APHIDS_POP_TIMEOUT 1

#define APHIDS_ERR_GET_FAIL   -3
#define APHIDS_ERR_SET_FAIL   -4
#define APHIDS_ERR_PUT_FAIL   -5
#define APHIDS_ERR_POP_FAIL   -6

#define APHIDS_REDIS_HOST "127.0.0.1"
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

/* aphids_put, aphids_pop

   These functions manipulate a queue assigned to an APHIDS key. Similar to the get/set
   commands, the keys have a prefix assigned to them that depends on the thread
   name and hashpipe instance ID (see the get/set documentation). The queue is a
   First In First Out (FIFO) type queue.

   NOTE: the pop command will block if the queue is empty unless timeout is >0.

*/
int aphids_put(aphids_context_t * aphids_ctx, char * key, char * value);
int aphids_pop(aphids_context_t * aphids_ctx, char * key, char * value, int timeout);

#endif
