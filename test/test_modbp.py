from context import modbp


def main():

    bpgc=modbp.BP_GraphColor(n=100,p=3.0/100,q=4,beta=1)
    bpgc.run()
    bpgc.print_marginals(limit=10)
    return 0

if __name__=='__main__':
    main()