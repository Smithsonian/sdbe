#!/bin/sh

# Usage: $0 expr [rate [true|false]]
# e.g.      vObo [125000  [true]]

cd $HOME/difx/data

sm="12 34"
save=${3-'true'}
rate=${2-'125000'}
expr=$1

for m in $sm
do
    fusermount -u ./mnt$m
    $save || rm -f mod-$m-$expr
    [ -f mod-$m-$expr ] && {
        echo reusing mod-$m-$expr
        vdifuse -u mod-$m-$expr ./mnt$m
        true
    } || {
        echo creating mod-$m-$expr
        eval vdifuse -a mod-$m-$expr -xm6sg -xrate=$rate -xinclpatt=$expr \
            ./mnt$m /mnt/disks/[$m]/?/data
    }
    ls -l mod-$m-$expr
done

# Q?? are links to the 4 quadrants on ALMA
#[ -n "`ls Q??`" ] &&
#    ls -lh Q??/se*/*/??/*.vdif && exit 0

# other standard setups, perhaps here.

# this should always work
match=`echo $expr | sed 'y/_/\//'`
#echo $match
ls -lh ./mnt??/sequences/${match}.vdif

# eof
