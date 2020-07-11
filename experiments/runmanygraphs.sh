for gamma in 0.5 1 1.5 
do
    for omega in 0 1 2
    do 
        for eta in $(seq 0.0 0.05 1.01)
        do
            for eps in $(seq 0.0 0.05 1.01)
            do
                #echo "python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
                sbatch -t 240 -n 1 -o test.out -p general --wrap="python rungraph.py 512 2 40 ${eta} 16 ${eps} 100 ${omega} ${gamma}"
            done
        done
    done
done