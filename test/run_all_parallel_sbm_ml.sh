for gamma in 0.5 1 1.5 3
do
    for omega in 0 1 2 4
    do
        for eta in $(seq 0.0 0.04 1.00)
        do
            for eps in $(seq 0.0 0.04 1.00)
            do
                #echo "python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
                sbatch -t 1000 -n 1 -o run_sbm_many.out -p general --wrap="python run_sbm_ml_test.py 500 2 40 ${eta} 16 ${eps} 50 ${omega} ${gamma}"
            done
        done
    done
done
