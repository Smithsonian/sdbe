#!/usr/bin/bash
CPU_IN=1
CPU_OUT=2
hashpipe -p easy_thread -c $CPU_IN easy_in_thread -c $CPU_OUT easy_out_thread > output.log
