for gamma in seq 0 .5 8
do
    for omega in 0 .5 8
    do

        #echo "python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
        sbatch -t 200 -n 1 -o senate_run.out -p general --wrap="python run_modbp_senate.py.py ${gamma} ${omega}"

    done
done
