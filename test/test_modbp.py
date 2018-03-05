from context import modbp


def main():
    rer=modbp.RandomERGraph(n=100,p=.4)
    adj=rer.get_adjacency()
    el=rer.get_edgelist()
    return 0

if __name__=='__main__':
    main()