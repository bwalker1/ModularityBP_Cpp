for gamma in `seq 0.001 .025 3`
do
    for omega in `seq 0 .05 8`
    do
        sbatch -t 200 -n 1 -o senate_run.out -p general --wrap="python3 run_modbp_senate.py ${gamma} ${omega}"
    done
done
