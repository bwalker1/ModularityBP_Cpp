
for eps in $(seq 0.0 0.025 .351)
    do
    for gamma in .5 1 1.5
        do
            sbatch -t 100 -n 1 -o /nas/longleaf/home/wweir/test_lfrrun2.txt -p general \
            --wrap "python /nas/longleaf/home/wweir/ModBP_proj/test run_LFR_test_with_sbmbp.py\
             1000 4 ${eps} 3 ${gamma} 50"
        done
    done