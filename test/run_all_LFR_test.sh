
for eps in $(seq 0.0 0.005 .4)
    do
    for gamma in .5 .75 1 1.25 1.5
        do
	    echo "${ep} ${gamma}" 
            sbatch -t 100 -n 1 -o /nas/longleaf/home/wweir/test_lfrrun2.txt -p general \
            --wrap "python /nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/run_LFR_test_with_sbmbp.py\
             1000 4 ${eps} 3 ${gamma} 50"
        done
    done
