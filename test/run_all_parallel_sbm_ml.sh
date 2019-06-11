for gamma in 1 1.5
do
    for omega in 0
    do
        for eta in $(seq 0.0 0.02 1.00)
        do
            for eps in $(seq 0.0 0.02 .2)
            do
                #echo "python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
                sbatch -t 600 -n 1 -o run_sbm_many.out -p general --wrap="python run_sbm_ml_test.py 250 2 20 ${eta} 10 ${eps} 50 ${omega} ${gamma}"
            done
        done
    done
done
