gammas=( 1.0 )
omegas=( 0  1.000e-03, 3.72e-03, 1.389e-02, 5.179e-02, 1.931e-01, 7.197e-01, 2.682e+00, 1.000e+01 )
#gammas=(1.0)
#omegas=(1.0)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for p in $(.5 .85 .95 .99 1.0)
            do
            for mu in $(seq 0 1 .1)
                do
	            #echo "${eps} ${gamma} ${omega}"
                sbatch -t 1000 -n 1 -o /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/test_mulitlayer_runs.txt -p general \
                --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/run_multilayer_matlab_test.py\
                1000 15 ${mu} ${p} ${omega} ${gamma} 100"
                done
            done
        done
    done
