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
#include <random>
#include <vector>
#include <time.h>
#include "bp.h"

using namespace std;

int main(int argc, const char * argv[]) {
    // simple testing
    index_t n = 500;
    index_t nt = 2;
    
    vector<index_t> layer_membership(n);
    for (index_t i=0;i<n;++i)
    {
        if (i < n/2)
        {
            layer_membership[i] = 0;
        }
        else
        {
            layer_membership[i] = 1;
        }
    }
    
    // create intra edgelist
    double c = 10;
    uniform_real_distribution<double> pdist(0,1);
    default_random_engine rng(time(NULL));
    double p = c/(n/2 - 1);
    
    vector<pair<index_t, index_t> > intra_edgelist;
    vector<pair<index_t, index_t> > inter_edgelist;
    
    for (index_t i=0;i<n/2;++i)
    {
        for (index_t j=0;j<n/2;++j)
        {
            if (i==j) continue;
            
            if (pdist(rng) < p)
            {
                intra_edgelist.push_back(pair<index_t,index_t>(i,j));
            }
            if (pdist(rng) < p)
            {
                intra_edgelist.push_back(pair<index_t,index_t>(i+n/2,j+n/2));
            }
        }
        inter_edgelist.push_back(pair<index_t,index_t>(i,i+n/2));
    }
    
    int q = 4;
    double omega = 1;
    
    
    //BP_Modularity(const vector<index_t> &layer_membership, const vector<pair<index_t, index_t> > &intra_edgelist, const vector<pair<index_t, index_t> > &inter_edgelist, const index_t _n, const index_t _nt, const int q, const double beta, const double omega = 1.0, const double resgamma = 1.0, bool verbose = false, bool transform = false);
    BP_Modularity bp(layer_membership,intra_edgelist,inter_edgelist, n,nt,q,1,omega,1.0,false,false);
    printf("Starting computation\n");
    clock_t start = clock();
    bp.run(100);
    clock_t finish = clock()-start;
    printf("%f seconds elapsed\n",double(finish)/double(CLOCKS_PER_SEC));
}
