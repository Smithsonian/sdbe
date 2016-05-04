#!/bin/bash

# This script searches for scans that match the given exp/obs/scan 
# search strings, and for each matched item calls verify_scan.py on 
# that scan. Matches are found against aphids scatter-gather files
# For March2016: ./batch_verify.sh e16b08 Sm "099-07*" 1 "/home/ayoung/Work/obs/March2016/verify-out"
if [ "$#" -lt 5 ] ; then
	echo "Usage: $0 EXP OBS SCAN QUAD LOGPATH" >&2
	exit 1
fi

EXP=$1
OBS=$2
SCAN=$3
QUAD=$4
LOGPATH=$5

EXEC="./verify_scan.py"
ARGS="-c 256 -v 1 -q ${QUAD}"
BATCH_LOG="${LOGPATH}/batch_verify_q${QUAD}.log"

if [ -w $BATCH_LOG ] ; then
	echo "Batch started `date`" >> $BATCH_LOG
else 
	echo "Batch started `date`" > $BATCH_LOG
fi
echo "====================" >> $BATCH_LOG

#~ PATH_SINGLE_DISH=/home/ayoung/Work/obs/March2015/single-dish-data
#~ SEARCH_STR=`echo "${PATH_SINGLE_DISH}/${EXP}_${OBS}_${SCAN}.vdif"`
#~ for f in `ls $SEARCH_STR` ; do
PATH_APHIDS_DATA="/mnt/disks/1/0/data/swarm_q${QUAD}"
SEARCH_STR=`echo "${PATH_APHIDS_DATA}/${EXP}_${OBS}_${SCAN}.vdif"`
for f in `ssh Mark6-4016 "ls $SEARCH_STR"` ; do
	FILENAME=`echo $f | grep -E -o "[^[:space:]/]*.vdif"`
	PARTS=`echo $FILENAME | sed y/\_/\ / | sed s/".vdif"//`
	if $EXEC $ARGS $PARTS ; then 
		echo "SUCCESS ($FILENAME)" >> $BATCH_LOG
	else
		if [ $? -eq 2 ] ; then 
			# happens when correlation test fails
			echo "FAILURE ($FILENAME)" >> $BATCH_LOG
		elif [ $? -eq 1 ] ; then
			# happens when flat-file could not be created/copied
			echo "ERROR ($FILENAME)" >> $BATCH_LOG
		fi
	fi
done
