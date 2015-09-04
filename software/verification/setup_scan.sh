#!/bin/bash

# This script prep-one-scans the given scan name, dds a section of data
# from the start of the scan.
if [ "$#" -lt 3 ] ; then
	echo "Usage: $0 EXP OBS SCAN" >&2
	exit 1
elif [ `ls /mnt/disks/?/?/data/${1}_${2}_${3}.vdif | grep -c .vdif` -ne 32 ] ; then
	echo "Invalid scan name (no such file)"
	exit 1
fi

EXP=$1
OBS=$2
SCAN=$3
PKTCOUNT=${4-"1024"}
PKTSIZE=${5-"4128"}
POS_PATH="/home/oper/difx/data"
POS_EXEC="/home/oper/difx/data/prep-one-scan.sh"

# change to prep-one-scan required path
SCANNAME="${EXP}_${OBS}_${SCAN}"
PWD_OLD=$PWD
cd $POS_PATH
rm mod-??-*
$POS_EXEC $SCANNAME

# change to original path and dd a slice
cd $PWD_OLD
for TAG in "12" "34" ; do
	dd if=$POS_PATH/mnt$TAG/sequences/$EXP/$OBS/$SCAN.vdif of=$PWD_OLD/${EXP}_${OBS}_${SCAN}_aphids-${TAG}.vdif bs=$PKTSIZE count=$PKTCOUNT
done
