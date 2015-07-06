#ifndef SGCOMM_REPORT_H
#define SGCOMM_REPORT_H

typedef enum {
	RL_DEBUGVVV,
	RL_DEBUG,
	RL_INFO,
	RL_NOTICE,
	RL_WARNING,
	RL_ERROR,
	RL_SEVERE,
	RL_NONE
} report_level;

typedef enum {
	RLT_SYSLOG,
	RLT_FILE,
	RLT_STDOUT,
	RLT_NONE
} report_log_type;

typedef struct status_update_struct {
	char **paths;
	char **names;
	char **values;
	int n;
} status_update;

/* Text-based logging */
void open_logging(report_level rl, report_log_type rlt, const char *name);
void log_message(report_level rl, const char *fmt, ...);
void vlog_message(report_level rl, const char *fmt, va_list ap);
void close_logging(void);

/* Interface to REDIS server */
void open_redis(void);
void redis_message(const char *path, const char *name, const char *value);
void close_redis(void);

#endif // SGCOMM_REPORT_H
