from context import modbp


def main():
    n=10000
    bpgc=modbp.BP_GraphColor(n=n,p=3.0/n,q=4,beta=1)
    bpgc.run()
    bpgc.print_marginals(limit=10)
    return 0

if __name__=='__main__':
    main()