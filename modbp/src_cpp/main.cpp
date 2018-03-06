//
//  main.cpp
//  ModularityBP
//
//  Created by Benjamin Walker on 3/2/18.
//  Copyright © 2018 Benjamin Walker. All rights reserved.
//

#include <iostream>
#include "bp.h"

int main(int argc, const char * argv[]) {
    index_t n = 1e5;
    BP_Modularity bp(n,3.0/n,4,1);
    
    clock_t start = clock();
    bp.run();
    clock_t finish = clock();
    bp.print_marginals(100);
    printf("%f seconds elapsed.\n",double(finish-start)/CLOCKS_PER_SEC);
}
