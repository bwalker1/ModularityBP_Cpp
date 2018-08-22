gammas=( .5 1.0 1.5)
omegas=(0 .1 .2 .4)
#gammas=(1.0)
#omegas=(1.0)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for eta in $(seq 0.8 0.2 1.0)
            do
            for eps in $(seq 0.0 0.1 .6)
                do
	            echo "${eps} ${gamma} ${omega}"
                sbatch -t 100 -n 1 -o /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/test_mulitlayer_runs.txt -p general \
                --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/run_multilayer_matlab_test.py\
                250 5 10 ${eps} ${eta} ${omega} ${gamma} 10"
                done
            done
        done
    done
