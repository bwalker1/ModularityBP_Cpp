//
//  main.cpp
//  Mod_BP_Xcode
//
//  Created by Benjamin Walker on 12/12/19.
//  Copyright © 2019 Benjamin Walker. All rights reserved.
//

#include <stdio.h>
#include "bp.h"
#include <random>

int main(void)
{
    // create all the stuff that defines a simple graph
    
    // perhaps just some random edges or something
    index_t n = 1000;
    
    vector<vector<index_t>> layer_membership; // NOTE: tentatively not initializing this - seems like only currently used in permute_beliefs function
    layer_membership.resize(n);
    vector<pair<index_t, index_t> > intra_edgelist;
    vector<pair<double,double>> intra_edgeweight;
    vector<double> inter_edgeweight;
    vector<pair<index_t, index_t> > inter_edgelist;

	vector<int> cluster_membership;
	cluster_membership.resize(n);
	for (int i = 0; i < n; ++i)
	{
		int c;
		if (i > 750)
		{
			c = 0;
		}
		else if (i > 500)
		{
			c = 1;
		}
		else if (i > 250)
		{
			c = 1;
		}
		else
		{
			c = 0;
		}
		cluster_membership[i] = c;
	}
    
    for (int i=0;i<n;++i)
    {
        for (int j=i+1;j<n;++j)
        {
			double p = cluster_membership[i] == cluster_membership[j] ? 0.5 : 0.001;
            if (double(rand())/double(RAND_MAX) < p)
            {
				//printf("Edge from %d to %d\n", i, j);
                intra_edgelist.push_back(pair<index_t, index_t>(i, j));
                intra_edgeweight.push_back(pair<double, double>(0, 1));
            }
        }
    }
    

    index_t nlayers = 1;
    int q = 4;
    index_t num_biparte_classes = 0;
    double beta = 1;
    vector<index_t> bipartite_class;
    double omega = 1.0;
    double dumping_rate = 1.0;
    double resgamma = 1.0;
    bool verbose = false;
    bool transform = false;
	bool parallel = false;
    
    // create the bp class
    auto bp = BP_Modularity(layer_membership, intra_edgelist, intra_edgeweight, inter_edgeweight, inter_edgelist, n, nlayers, q, num_biparte_classes, beta, bipartite_class, omega, dumping_rate, resgamma, verbose, transform, parallel);
    
    for (int i=0;i<100; ++i)
    {
        bp.step();
    }

	for (auto v : bp.return_marginals())
	{
		for (auto d : v)
		{
			printf("%f ", d);
		}
		printf("\n");
	}
}
