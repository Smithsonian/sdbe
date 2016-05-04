#!/bin/bash

# This script does setup_scan.sh on remote machine and then copies the
# result to the current path
if [ "$#" -lt 4 ] ; then
	echo "Usage: $0 EXP OBS SCAN QUAD [ PKTCOUNT [ PKTSIZE [ OUTPATH ] ] ]" >&2
	exit 1
fi

EXP=$1
OBS=$2
SCAN=$3
QUAD=$4
PKTCOUNT=${5-"1024"}
PKTSIZE=${6-"8224"}
OUTPATH=${7-"./"}

# do setup_scan.sh
REMOTE_HOST="Mark6-4016"
REMOTE_USER="oper"
REMOTE_PATH="/home/oper/ayoung"

if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" ". /home/${REMOTE_USER}/.profile ; cd ${REMOTE_PATH} ; ./setup_scan.sh ${EXP} ${OBS} ${SCAN} ${QUAD} ${PKTCOUNT} ${PKTSIZE} " ; then
	exit 1
fi

# copy data: APHIDS from Mark6
for TAG in "12" "34" ; do
	if ! scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${EXP}_${OBS}_${SCAN}_aphids-${TAG}_q${QUAD}.vdif" "${OUTPATH}" ; then
		exit 1
	fi
done

# do get_snippet.sh (which copies data too)
REMOTE_HOST="hamster"
REMOTE_USER="ayoung"
REMOTE_PATH="/data1/March2016/single-dish-data"
if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" ". /home/${REMOTE_USER}/.profile ; cd ${REMOTE_PATH} ; ./get_snippet.sh ${EXP} ${OBS} ${SCAN} ${QUAD} ${PKTCOUNT} ${PKTSIZE} " ; then
	exit 1
fi
