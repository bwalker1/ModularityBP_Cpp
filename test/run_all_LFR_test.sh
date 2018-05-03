gammas=( .5 1.0 1.5)
for eps in $(seq 0.0 0.02 .4)
    do
    for gamma in "${gammas[@]}"
        do
	    echo "${eps} ${gamma}"
            sbatch -t 500 -n 1 -o /nas/longleaf/home/wweir/test_lfrrun2.txt -p general \
            --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/run_LFR_test_with_sbmbp.py\
             1000 4 ${eps} 3 ${gamma} 100" >> submitted.txt
        done
    done
