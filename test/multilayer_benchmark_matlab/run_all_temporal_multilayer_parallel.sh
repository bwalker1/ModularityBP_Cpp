#! /usr/bin/bash
gammas=( 1.0 )
omegas=( 0  1.000e-03 3.72e-03 1.389e-02 5.179e-02 1.931e-01 .3 7.197e-01 1.0 2.682e+00 1.000e+01 )
ps=( .5 .85 .95 .99 1.0 )
#ps=( .99 )
mus=(`seq 0 .1 1`)
#mus=( .1 .2 )
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for p in "${ps[@]}"
            do
            for mu in "${mus[@]}"
                do
	            #echo "${eps} ${gamma} ${omega}"
                sbatch -t 4000 -n 1 -o /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/test_multilayer_temporal.txt -p general \
                --wrap "python3 /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/run_multilayer_temporal_test.py\
                150 100 ${mu} ${p} ${omega} ${gamma} 100"
                done
            done
        done
    done
