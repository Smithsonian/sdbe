#include <syslog.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include "sgcomm_report.h"

#define DEFAULT_LOG_FILE_NAME "sgcomm_report.log"
#define DEFAULT_LOG_FILE_MODE "a+"
#define DEFAULT_LOG_SYSLOG_OPTION (LOG_CONS | LOG_PID)
#define DEFAULT_LOG_SYSLOG_FACILITY (LOG_INFO)
#define TIMESTAMP_FORMAT_LONG "%Y-%m-%d %H:%M:%S"
#define TIMESTAMP_FORMAT_SHORT "%H:%M:%S"

#define MAX_MSG_LEN 0x100
#define MAX_TIMESTAMP_LEN 0x100

//~ static char _msg[MAX_MSG_LEN];

static report_level _log_level = RL_NONE;
static report_log_type _log_type = RLT_NONE;
static int _log_initialized = 0;
static FILE *_log_fd = NULL;

static void _log_file(report_level, const char *fmt, va_list ap);
static void _log_syslog(report_level, const char *fmt, va_list ap);
static void _set_log_level(report_level rl);
static void _set_log_type(report_log_type rlt, const char *name);
static const char * _get_level_tag(report_level rl);
static void _get_timestamp_long(char *time_str);
static void _get_timestamp_short(char *time_str);

void open_logging(report_level rl, report_log_type rlt, const char *name) {
	_set_log_level(rl);
	_set_log_type(rlt,name);
}

void close_logging(void) {
	char _tsmsg[MAX_TIMESTAMP_LEN];
	switch(_log_type) {
	case RLT_SYSLOG:
		closelog();
		break;
	case RLT_FILE:
		if (_log_fd != NULL) {
			_get_timestamp_short(_tsmsg);
			fprintf(_log_fd,"Logging closed %s.\n===============================\n",_tsmsg);
			fclose(_log_fd);
		}
		break;
	case RLT_STDOUT:
		break;
	default:
		break;
	}
	_set_log_level(RL_NONE);
	_set_log_type(RL_NONE,NULL);
}

void log_message(report_level rl, const char *fmt, ...) {
	va_list ap;
	
	va_start(ap,fmt);
	vlog_message(rl, fmt, ap);
	va_end(ap);
}

void vlog_message(report_level rl, const char *fmt, va_list ap) {
	if (rl < _log_level)
		return;
	if (fmt == NULL) {
		return;
	}
	switch(_log_type) {
	case RLT_SYSLOG:
		_log_syslog(rl,fmt,ap);
		break;
	case RLT_FILE:
	case RLT_STDOUT:
		_log_file(rl,fmt,ap);
		break;
	default:
		break;
	}
}

void open_redis(void) {
	// TODO: open connection, set globals, etc
}

void redis_message(const char *path, const char *name, const char *value) {
	// TODO: send message
}

void close_redis(void) {
	// TODO: clean-up
}

void _set_log_level(report_level rl) {
	_log_level = rl;
}

void _set_log_type(report_log_type rlt, const char *name) {
	char _tsmsg[MAX_TIMESTAMP_LEN];
	_log_type = rlt;
	switch(rlt) {
	case RLT_SYSLOG:
		openlog(name,DEFAULT_LOG_SYSLOG_OPTION,DEFAULT_LOG_SYSLOG_FACILITY);
		break;
	case RLT_FILE:
		if (name == NULL)
			_log_fd = fopen(DEFAULT_LOG_FILE_NAME,DEFAULT_LOG_FILE_MODE);
		else
			_log_fd = fopen(name,DEFAULT_LOG_FILE_MODE);
		if (_log_fd != NULL) {
			_get_timestamp_long(_tsmsg);
			fprintf(_log_fd,"===============================\nLogging opened %s\n",_tsmsg);
		}
		break;
	case RLT_STDOUT:
		_log_fd = stdout;
		break;
	}
}

static void _log_syslog(report_level rl, const char *fmt, va_list ap) {
	int priority;
	switch(rl) {
	case RL_DEBUGVVV:
	case RL_DEBUG:
		priority = LOG_DEBUG;
		break;
	case RL_INFO:
		priority = LOG_INFO;
		break;
	case RL_NOTICE:
		priority = LOG_NOTICE;
		break;
	case RL_WARNING:
		priority = LOG_WARNING;
		break;
	case RL_ERROR:
		priority = LOG_ERR;
		break;
	case RL_SEVERE:
		priority = LOG_CRIT;
		break;
	default:
		priority = 0;
		break;
	}
	vsyslog(priority, fmt, ap);
}

static void _log_file(report_level rl, const char *fmt, va_list ap) {
	char _tsmsg[MAX_TIMESTAMP_LEN];
	char _msg[MAX_MSG_LEN];
	if (_log_fd < 0)
		return;
	_get_timestamp_short(_tsmsg);
	//~ fprintf(_log_fd,"[%s]",_tsmsg);
	vsnprintf(_msg, MAX_MSG_LEN, fmt, ap);
	fprintf(_log_fd,"[%s][%s]:%s\n",_tsmsg,_get_level_tag(rl),_msg);
	fflush(_log_fd);
}

const char * _get_level_tag(report_level rl) {
	switch(rl) {
	case RL_DEBUGVVV:
		return "DEBUGVVV";
	case RL_DEBUG:
		return "DEBUG";
	case RL_INFO:
		return "INFO";
	case RL_NOTICE:
		return "NOTICE";
	case RL_WARNING:
		return "WARNING";
	case RL_ERROR:
		return "ERROR";
	case RL_SEVERE:
		return "SEVERE";
	default:
		return "INFO";
	}
}

static void _get_timestamp_long(char *time_str) {
	time_t t;
	struct tm timestamp;
	
	time(&t);
	localtime_r(&t,&timestamp);
	strftime(time_str,MAX_TIMESTAMP_LEN,TIMESTAMP_FORMAT_LONG,&timestamp);
}

static void _get_timestamp_short(char *time_str) {
	time_t t;
	struct tm timestamp;
	
	time(&t);
	localtime_r(&t,&timestamp);
	strftime(time_str,MAX_TIMESTAMP_LEN,TIMESTAMP_FORMAT_SHORT,&timestamp);
}
