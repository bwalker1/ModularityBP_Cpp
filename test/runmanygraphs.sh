rm "~/data/*"
for eta in $(seq 0.0 0.05 1.01)
do
    for eps in $(seq 0.0 0.05 1.01)
    do
        sbatch -t 60 -n 1 -o test.out -p general --wrap="python rungraph.py 1000 2 10 ${eta} 5 ${eps} 100 1"
    done
done