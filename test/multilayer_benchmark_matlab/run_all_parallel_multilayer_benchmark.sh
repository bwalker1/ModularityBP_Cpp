#gammas=( .5 1.0 1.5)
#omegas=(0 1.0 2.0 3.0)
gammas=(1.0)
omegas=(1.0)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for eta in $(seq 0.0 0.04 .6)
            do
            for eps in $(seq 0.0 0.04 .6)
                do
	            echo "${eps} ${gamma} ${omega}"
                sbatch -t 20 -n 1 -o /nas/longleaf/home/wweir/test/multilayer_benchmark_matlab/test_mulitlayer_runs.txt -p general \
                --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/run_multilayer_matlab_test.py\
                500 5 10 ${eps} ${eta} ${omega} ${gamma} 2"
                done
            done
        done
    done
