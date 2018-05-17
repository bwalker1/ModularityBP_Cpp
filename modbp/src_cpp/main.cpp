//
//  main.cpp
//  ModularityBP
//
//  Created by Benjamin Walker on 3/2/18.
//  Copyright © 2018 Benjamin Walker. All rights reserved.
//

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "bp.h"

using namespace std;

int main(int argc, const char * argv[]) {
    index_t n = 0;
    
    int q = 5;
    double omega = 1;
    //BP_Modularity(const vector<index_t> &layer_membership, const vector<pair<index_t, index_t> > &intra_edgelist, const vector<pair<index_t, index_t> > &inter_edgelist, const index_t _n, const index_t _nt, const int q, const double beta, const double omega = 1.0, const double resgamma = 1.0, bool verbose = false, bool transform = false);
    BP_Modularity bp(vector<index_t>(),vector<pair<index_t,index_t> >(),vector<pair<index_t,index_t> >(), n,1,q,1,omega,1.0,false,false);
    printf("Starting computation\n");
    clock_t start = clock();
    //printf("%f\n",bp.compute_bstar());
    clock_t finish = clock()-start;
    printf("%f seconds elapsed\n",double(finish)/double(CLOCKS_PER_SEC));
}
