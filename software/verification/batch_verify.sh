#!/bin/bash

# This script searches for scans that match the given exp/obs/scan 
# search strings, and for each matched item calls verify_scan.py on 
# that scan. Matches are found against pre-processed VDIF output.
if [ "$#" -lt 3 ] ; then
	echo "Usage: $0 EXP OBS SCAN [WORK_DIR [QUAD SIDEBAND RXBAND [R2INP APHINP]]]

Mandatory arguments:
  EXP        Expriment name
  OBS        Observatory name (typically 'Sw')
  SCAN       Scan name (typically jjj-HHMM date format)
Optional arguments:
  WORK_DIR   Root path to where input / output files are located,
             default is './work'.
  QUAD       SWARM quadrant where data came from
  SIDEBAND   Receiver sideband corresponding to data
  RXBAND     Receiver band for observation
  R2INP      Input on the R2DBE datastream
  APHINP     Input on the APHIDS datastream

The program expects the following directories in WORK_DIR:
  WORK_DIR/sched - should contain schedule in .xml format that
                   matches the experiment name
  WORK_DIR/ref-ant - should contain VDIF data for the single dish
                     (redundant) antenna
Output produced by the program are also stored in WORK_DIR:
  WORK_DIR/batch_verify.log - contains a log of all calls to
                              $0
  WORK_DIR/array - contains fragments of pre-processed array VDIF
                   copied over for the verification
" >&2
	exit 1
fi

# Scan matching parameters
EXP=$1
OBS=$2
SCAN=$3

# Optional parameters that set paths to input / ouput files
WORK_DIR=${4-"./work"}

# Optional parameters that set the quadrant, sideband and receiver band
QUAD=${5-"1"}
SIDEBAND=${6-"USB"}
RXBAND=${7-"230"}

# Optional parameters that select inputs
R2INP=${8-"1"}
APHINP=${9-"1"}

# Check the required input directories exist
SCHED_DIR=${WORK_DIR}/sched
if ! [ -d ${SCHED_DIR} ] ; then
	echo "No schedule directory found, expected '${SCHED_DIR}'"
	exit 1
fi
SCHED_FILE=${SCHED_DIR}/${EXP}.xml
if ! [ -f ${SCHED_FILE} ] ; then
	echo "No schedule found, expected '${SCHED_FILE}'"
	exit 1
fi
SINGLE_DISH_DIR=${WORK_DIR}/ref-ant
if ! [ -d ${SINGLE_DISH_DIR} ] ; then
	echo "No single-dish data directory found, expected '${SINGLE_DISH_DIR}'"
	exit 1
fi

# Create necessary output directories
ARRAY_DIR=${WORK_DIR}/array
mkdir -p ${ARRAY_DIR}
LOG_DIR=${WORK_DIR}/log
mkdir -p ${LOG_DIR}

EXEC="./verify_scan.py"
ARGS="-c 1024 -v 1 --quad ${QUAD} --sideband ${SIDEBAND} --frequency-band ${RXBAND} --aphids-input ${APHINP} --r2dbe-input ${R2INP}"
BATCH_LOG="${WORK_DIR}/batch_verify.log"

echo "Batch started `date`" >> $BATCH_LOG
echo "====================" >> $BATCH_LOG

HOST_SWARM_DATA="Mark6-4016"
PATH_SWARM_DATA=/mnt/disks/1/0/data
SEARCH_STR=`echo "${PATH_SWARM_DATA}/${EXP}_${OBS}_${SCAN}.vdif"`
for f in `ssh ${HOST_SWARM_DATA} "ls ${SEARCH_STR}"` ; do
	FILENAME=`echo $f | grep -E -o "[^[:space:]/]*.vdif"`
	PARTS=`echo ${FILENAME} | sed y/\_/\ / | sed s/".vdif"//`
	echo ${EXEC} ${ARGS} ${PARTS}
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
