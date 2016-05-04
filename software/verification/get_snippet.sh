#!/bin/bash
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

if [ $QUAD -eq 1 ] ; then
	SRC="/data1/March2016/single-dish-data/mnt34-e/sequences/${EXP}/${OBS}/${SCAN}.vdif"
	DST="/data1/March2016/single-dish-data/${EXP}_${OBS}_${SCAN}_r2dbe_q1.vdif"
elif [ $QUAD -eq 2 ] ; then
        SRC="/data1/March2016/single-dish-data/mnt12-e/sequences/${EXP}/${OBS}/${SCAN}.vdif"
        DST="/data1/March2016/single-dish-data/${EXP}_${OBS}_${SCAN}_r2dbe_q2.vdif"
fi
dd if=$SRC of=$DST bs=$PKTSIZE count=$PKTCOUNT
echo "File ready:"
ls $DST
echo "Copying..."
scp $DST ayoung@barrett:~/Work/obs/March2016/single-dish-data/q${QUAD}
echo "Clean-up..."
rm $DST
