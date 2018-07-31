for gamma in `seq 0 .1 3`
do
    for omega in `seq 0 .2 8`
    do
        echo "sbatch -t 400 -n 1 -o senate_run.out -p general --wrap=\"python run_modbp_senate.py ${gamma} ${omega}\""
        sbatch -t 400 -n 1 -o senate_run.out -p general --wrap="python run_modbp_senate.py ${gamma} ${omega}"

    done
done
