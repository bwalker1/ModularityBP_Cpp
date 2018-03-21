#!/bin/bash
for beta in $(seq 0.5 0.05 3.0)
do
    res=`mod infer -l $1 -q2 -b $beta -v2`
    #echo "$beta,$res"
    #echo $res
    retmod=`echo "$res" | grep retrieval_modularity | cut -d'=' -f 2`
    iters=`echo "$res" | grep iter_time | cut -d'=' -f 2`
    echo "$beta,$iters"
done
