#!/bin/bash

# This script does setup_scan.sh on remote machine and then copies the
# result to the current path
if [ "$#" -lt 3 ] ; then
	echo "Usage: $0 EXP OBS SCAN" >&2
	exit 1
fi

EXP=$1
OBS=$2
SCAN=$3
PKTCOUNT=${4-"1024"}
PKTSIZE=${5-"8224"}
OUTPATH=${6-"./work/array"}

REMOTE_HOST="Mark6-4016"
REMOTE_USER="oper"
REMOTE_PATH="/home/oper/tmp"

# do setup_scan.sh
if ! ssh "${REMOTE_USER}@${REMOTE_HOST}" ". /home/${REMOTE_USER}/.profile ; cd ${REMOTE_PATH} ; ./setup_scan.sh ${EXP} ${OBS} ${SCAN} ${PKTCOUNT} ${PKTSIZE} " ; then
	exit 1
fi

# copy data
for TAG in "12" "34" ; do
	if ! scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${EXP}_${OBS}_${SCAN}_aphids-${TAG}.vdif" "$OUTPATH" ; then
		exit 1
	fi
done
