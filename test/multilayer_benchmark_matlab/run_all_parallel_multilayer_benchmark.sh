gammas=( .5 1.0 1.5)
omegas=(0 1.0 2.0 3.0)
orig_layers=(0 1 2 3)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for eta in $(seq 0.0 0.02 .6)
            do
            for eps in $(seq 0.0 0.02 .6)
                do
	            echo "${eps} ${gamma} ${omega} ${orig_layer_ind}"
                sbatch -t 500 -n 1 -o /nas/longleaf/home/wweir/test/multilayer_benchmark_matlab/test_mulitlayer_runs.txt -p general \
                --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab/run_multilayer_matlab_test.py\
                500 5 10 ${eps} ${eta} ${gamma} ${omega} 2"
                done
            done
        done
    done
