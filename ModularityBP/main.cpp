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
    // insert code here..
    BP_GraphColor bp(1000000,5.0/1000000,4,1,false,false);
    clock_t start =clock();
    bp.run();
    printf("Running time: %f\n",double(clock()-start)/double(CLOCKS_PER_SEC));
    return 0;
}
