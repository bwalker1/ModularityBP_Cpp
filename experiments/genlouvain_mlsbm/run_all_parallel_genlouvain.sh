for gamma in 0.5 1 1.5
do
    for omega in 0 1 2
    do
        for eta in $(seq 0.001 0.04 1.00)
        do
            for eps in $(seq 0.001 0.04 1.00)
            do
                #echo "python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
                sbatch -t 50 -n 1 -o run_louvain_parallel.out -p general --wrap="python run_genlouvain_mlsbm.py 250 2 20 ${eta} 10 ${eps} 15 ${omega} ${gamma}"
            done
        done
    done
done
