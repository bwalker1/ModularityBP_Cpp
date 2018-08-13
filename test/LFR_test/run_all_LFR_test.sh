gammas=( .5 1.0 1.5)
omegas=(0 1.0 2.0 3.0)
orig_layers=(0 1 2 3)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for orig_layer_ind in "${orig_layers[@]}"
            do
            for eps in $(seq 0.0 0.02 .4)
                do
	            echo "${eps} ${gamma} ${omega} ${orig_layer_ind}"
                sbatch -t 500 -n 1 -o /nas/longleaf/home/wweir/test/LFR_test/test_lfrrun2.txt -p general \
                --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/run_LFR_test_with_sbmbp.py\
                500 ${eps} 6 ${gamma} 2 ${omega} ${orig_layer_ind}"
                done
            done
        done
    done
