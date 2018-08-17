gammas=( .5 1.0 1.5)
omegas=(0 1.0 2.0 3.0)
orig_layers=(0 1 2 3 4)
for omega in "${omegas[@]}"
    do
    for gamma in "${gammas[@]}"
        do
        for orig_layer_ind in "${orig_layers[@]}"
            do
            for eps in $(seq 0.0 0.02 .6)
                do
                echo "${eps} ${gamma} ${omega} ${orig_layer_ind}"
                sbatch -t 1000 -n 1 -o /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/LFR_test/test_lfrrun.txt -p general --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/LFR_test/run_LFR_test_with_sbmbp.py 500 ${eps} 10 ${gamma} 30 ${orig_layer_ind} ${omega}"
                done
            done
        done
    done
