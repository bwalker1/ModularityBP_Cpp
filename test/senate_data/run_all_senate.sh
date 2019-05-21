for gamma in `seq 0.001 .05 3`
do
    for omega in `seq 0 .05 12`
    do
        #echo "sbatch -t 800 -n 1 -o senate_run.out -p general --wrap=\"python run_modbp_senate.py ${gamma} ${omega}\""
        sbatch -t 1000 -n 1 -o senate_run.out -p general --wrap="python run_modbp_senate.py ${gamma} ${omega}"
    done
done
