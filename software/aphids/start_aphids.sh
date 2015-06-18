#!/usr/bin/env bash
CPU_IN=1
CPU_OUT=2
CPU_IN_OUT=3
hashpipe -p aphids -c $CPU_IN vdif_in_null_thread -c $CPU_IN_OUT vdif_inout_null_thread -c $CPU_OUT vdif_out_null_thread > output.log & 
