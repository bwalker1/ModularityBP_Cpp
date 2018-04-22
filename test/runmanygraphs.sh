
n=1000
q=2
nlayers=10


for eta in $(seq 0.0 0.05 1.01)
do
    for eps in $(seq 0.0 0.05 1.01)
    do
        echo "python rungraph.py 1000 2 10 ${eta} 5 ${eps} 100 1"
    done
done