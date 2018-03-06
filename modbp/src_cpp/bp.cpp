//
//  bp.cpp
//  beliefprop
//
//  Created by Benjamin Walker on 2/19/18.
//  Copyright © 2018 Benjamin Walker. All rights reserved.
//

#include "bp.h"
#include <vector>
#include <set>
#include <random>
#include <assert.h>
#include <math.h>
#include <stack>
#include <map>
#include <unordered_set>
using namespace std;


double truncate(const double in, const int q)
{
    return max(min(in,1.0),1.0/(5.0*q));
}

void print_array(index_t *arr, index_t n)
{
    for (int i=0;i<n;++i)
    {
        printf("%lu ",arr[i]);
    }
    printf("\n");
}
void print_array(double *arr, index_t n)
{
    for (int i=0;i<n;++i)
    {
        printf("%f ",arr[i]);
    }
    printf("\n");
}



BP_Modularity::BP_Modularity(const index_t _n, const double p, const int _q, const double _beta, bool _transform) : n(_n),q(_q), beta(_beta),  neighbor_count(_n), order(_n), rng((int)time(NULL)), transform(_transform), theta(_q)
{
    save = false;
    clock_t start = clock();
    
    printf("Constructing graph\n");
    
    scale = exp(beta)-1;
    eps = 1e-8;
    computed_marginals = false;
    // create a random Erdos-Renyi graph and set up the internal variables
    vector<vector<index_t> > edges(n);
    uniform_real_distribution<double> pdist(0,1);
    binomial_distribution<index_t> bdist(n,p/2);
    uniform_int_distribution<index_t> destdist(0,n-1);
    neighbor_offset_map.resize(n);
    num_edges = 0;
    for (index_t i=0;i<n;++i)
    {
        order[i] = i;
        //*
        // figure out how many edges to create
        index_t edges_to_create = bdist(rng);
        num_edges += edges_to_create;
        // find where the edges are going
        set<index_t> dest;
        while (dest.size() < edges_to_create)
        {
            auto val = destdist(rng);
            if (val == i) continue;
            dest.insert(val);
        }
        for (auto j : dest)
        {
            if (find(edges[i].begin(), edges[i].end(), j) != edges[i].end()) continue;
            edges[i].push_back(j);
            edges[j].push_back(i);
        }
    }
    
    num_edges *= 2;
    
    prefactor = -beta/num_edges;
    
    //*
    // perform isomorphic transform of graph to improve memory adjacency properties
    if (transform)
    {
        isomorphism.resize(n);
        r_isomorphism.resize(n);
        stack<index_t> to_process;
        for (index_t i=0;i<n;++i)
        {
            to_process.push(i);
        }
        unordered_set<index_t> processed;
        
        index_t iso_counter = 0;
        to_process.push(0);
        while (processed.size() < n)
        {
            auto val = to_process.top();
            to_process.pop();
            
            if (!processed.count(val))
            {
                r_isomorphism[iso_counter] = val;
                isomorphism[val] = iso_counter++;
                processed.insert(val);
                
                for (auto neighbor : edges[val])
                {
                    if (!processed.count(neighbor))
                    {
                        to_process.push(neighbor);
                    }
                }
            }
        }
        // apply the isomorphism
        vector<vector<index_t> > new_edges(n);
        for (int i=0;i<n;++i)
        {
            for (int j=0;j<edges[i].size();++j)
            {
                assert(0 <= isomorphism[i] && isomorphism[i] < n);
                new_edges[isomorphism[i]].push_back(isomorphism[edges[i][j]]);
            }
            sort(new_edges[isomorphism[i]].begin(),new_edges[isomorphism[i]].end());
        }
        edges = new_edges;
    }
    //*/
    
    beliefs = (double*) malloc(q*num_edges*sizeof(double));
    beliefs_new = (double*) calloc(q*num_edges,sizeof(double));
    beliefs_offsets = (size_t*) malloc((n+1)*sizeof(size_t));
    
    neighbors = (index_t*) malloc(num_edges*sizeof(index_t));
    neighbors_reversed = (index_t*) malloc(num_edges*sizeof(index_t));
    neighbors_offsets = (size_t*) malloc((n+1)*sizeof(size_t));
    
    marginals = (double*) malloc(q*n*sizeof(double));
    
    // set up offsets for fast access and copy graph structure into neighbors array
    size_t neighbors_offset_count = 0;
    size_t beliefs_offset_count = 0;
    int neighbor_c = 0;
    
    beliefs_offsets[0] = 0;
    neighbors_offsets[0] = 0;
    max_degree = 0;
    for (int i=0;i<n;++i)
    {
        beliefs_offset_count += q*edges[i].size();
        neighbors_offset_count += edges[i].size();
        neighbor_count[i] = edges[i].size();
        
        beliefs_offsets[i+1] = beliefs_offset_count;
        neighbors_offsets[i+1] = neighbors_offset_count;
        
        max_degree = max(max_degree,edges[i].size());
        
        for (int j=0;j<edges[i].size();++j)
        {
            assert(neighbor_c < num_edges);
            neighbors[neighbor_c++] = edges[i][j];
            neighbor_offset_map[i][edges[i][j]] = j;
        }
    }
    neighbor_c = 0;
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<edges[i].size();++j)
        {
            neighbors_reversed[neighbor_c++] = neighbor_offset_map[edges[i][j]][i];
        }
    }
    
    scratch = (double*) malloc(q*max_degree*sizeof(double));
    if (!scratch)
    {
        printf("Scratch failed to allocate\n");
        exit(1);
    }
    
    // set starting value of beliefs
    // generate values for each state and then normalize
    normal_distribution<double> eps_dist(0,0.1);
    exponential_distribution<double> unif_dist(1);
    //*
    for (size_t idx = 0;idx<q*num_edges;++idx)
    {
        double val = eps_dist(rng);
        beliefs[idx] = truncate(1.0/q + val,q);
        //beliefs[idx] = unif_dist(rng);
    }//*/

    for (index_t i=0;i<n;++i)
    {
        normalize(beliefs,i);
    }
    
    // initialize values of theta
    for (index_t i=0;i<n;++i)
    {
        index_t nn = n_neighbors(i);
        for (index_t s = 0; s<q;++s)
        {
            //theta[s] += nn * 1
        }
    }
    
    clock_t finish = clock();
    printf("Initialization: %f seconds elapsed.\n",double(finish-start)/double(CLOCKS_PER_SEC));
}

bool BP_Modularity::run()
{
    
    change = 1;
    unsigned long maxIters = 100;
    bool converged = false;
    for (unsigned long iter = 0; iter < maxIters; ++iter)
    {
        step();

        printf("Iteration %lu: change %f\n",iter+1,change);

        
        
        if (!changed)
        {
            converged = true;
            printf("Converged after %lu iterations.\n",iter+1);
            break;
        }
    }
    if (!converged)
    {
        printf("Algorithm failed to converge after %lu iterations.\n",maxIters);
    }
    
    return converged;
}

void BP_Modularity::step()
{
    changed = false;
    change = 0;
    
    // go through each node and update beliefs
    for (index_t node_idx = 0;node_idx<n;++node_idx)
    {
        index_t i = node_idx;
        const index_t nn = neighbor_count[i];
        if (nn==0) continue;
        
        // first, see how much change we had to incoming beliefs so we know if we need to update
        double local_change = 0;
        for (index_t idx=beliefs_offsets[i];idx<beliefs_offsets[i+1];++idx)
        {
            local_change += fabs(beliefs[idx] - beliefs_new[idx]);
        }
        local_change /= q*nn;
        if (local_change < eps)
        {
            // not enough change in incoming beliefs to warrant an update
            continue;
        }
        // if we changed any nodes, set this to true so we know we haven't converged
        changed = true;
        change += local_change;
        
        // update our record of what our incoming beliefs were for future comparison
        memcpy(beliefs_new+beliefs_offsets[i], beliefs+beliefs_offsets[i], q*nn*sizeof(double));

        
        // iterate over all states
        for (int s = 0; s < q;++s)
        {
            // incoming beliefs are already stored locally
            // figure out the sum of logs part of the update equation that uses the incoming beliefs
            for (index_t idx=0; idx<nn; ++idx)
            {
                scratch[nn*s+idx] = 0;
                for (index_t idx2 = 0;idx2<nn;++idx2)
                {
                    if (idx2 == idx) continue;
                    double add = log(1+scale*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
                    scratch[nn*s+idx] += add;
                }
                // evaluate the rest of the update equation
                scratch[nn*s+idx] = exp(prefactor*nn*theta[s] + scratch[nn*s+idx]);
            }
            
        }
        
        // normalize the scratch space
        for (index_t idx = 0;idx<nn;++idx)
        {
            // iterate over all states
            double sum = 0;
            for (size_t s = 0; s < q;++s)
            {
                sum += scratch[nn*s+idx];
            }
            if (sum > 0)
            {
                for (size_t s = 0; s < q;++s)
                {
                    scratch[nn*s+idx] /= sum;
                }
            }
            else
            {
                for (size_t s = 0; s < q;++s)
                {
                    scratch[nn*s+idx] = 1.0/q;
                }
            }
        }
        
        // write the scratch space out to non-local memory
        for (size_t s = 0; s < q; ++s)
        {
            for (index_t idx = 0; idx < nn; ++ idx)
            {
                index_t k = neighbors[neighbors_offsets[i]+idx];
                const index_t nnk = neighbor_count[k];
                index_t idx_out = neighbors_reversed[neighbors_offsets[i]+idx];
                
                beliefs[beliefs_offsets[k]+nnk*s+idx_out] = scratch[nn*s+idx];
            }
        }
    }
}

void BP_Modularity::normalize(double * beliefs, index_t i)
{
    const index_t nn = neighbor_count[i];
    // iterate over all neighbors
    for (size_t idx2 = 0; idx2 < nn; ++ idx2)
    {
        // iterate over all states
        double sum = 0;
        for (size_t s = 0; s < q;++s)
        {
            sum += beliefs[beliefs_offsets[i]+nn*s+idx2];
        }
        for (size_t s = 0; s < q;++s)
        {
            beliefs[beliefs_offsets[i]+nn*s+idx2] /= sum;
        }
    }
}

void BP_Modularity::print_beliefs()
{
    for (index_t i=0;i<n;++i)
    {
        index_t nn = neighbor_count[i];
        // iterate over all states
        for (int s = 0; s < q;++s)
        {
            for (index_t idx=0;idx<nn;++idx)
            {
                index_t k = neighbors[neighbors_offsets[i]+idx];
                double belief = beliefs[beliefs_offsets[i]+nn*s+idx];
                printf("X_%d^(%lu->%lu)=%f\n",s,i,k,belief);
            }
        }
    }
}


void BP_Modularity::compute_marginals()
{
    for (index_t node_idx=0;node_idx<n;++node_idx)
    {
        index_t i;
        if (transform)
        {
            i = r_isomorphism[node_idx];
        }
        else
        {
            i = node_idx;
        }
        double marginal_sum = 0;
        for (int s = 0; s < q; ++s)
        {
            double marginal = 1;
            const index_t nn = neighbor_count[i];
            for (index_t idx = 0; idx < nn;++idx)
            {
                marginal *= beliefs[beliefs_offsets[i] + nn*s + idx];
            }
            marginals[q*i + s] = marginal;
            marginal_sum += marginal;
        }
        for (int s=0; s < q;++s)
        {
            marginals[q*i + s] /= marginal_sum;
        }
    }
}

void BP_Modularity::print_marginals(size_t limit)
{
    compute_marginals();
    
    index_t print_count = 0;
    for (index_t i=0;i<n;++i)
    {
        for (int s=0;s<q;++s)
        {
            printf("P_%d(%lu) = %f\n",s,i,marginals[print_count++]);
        }
        if (print_count > limit)
        {
            return;
        }
    }
}

void BP_Modularity::print_beliefs(size_t limit)
{
    size_t print_count = 0;
    for (index_t i=0;i<n;++i)
    {
        index_t nn = neighbor_count[i];
        for (index_t idx=0;idx<nn;++idx)
        {
            index_t k = neighbors[neighbors_offsets[i]+idx];
            // iterate over all states
            for (int s = 0; s < q;++s)
            {
                
                
                double belief = beliefs[beliefs_offsets[i]+nn*s+idx];
                printf("X_%d^(%lu->%lu)=%f\n",s,i,k,belief);
                ++ print_count;
                if (print_count >= limit)
                {
                    return;
                }
            }
        }
    }
}

BP_Modularity::~BP_Modularity() { 
    free(beliefs);
    free(beliefs_new);
    free(beliefs_offsets);
    free(neighbors);
    free(neighbors_offsets);
    free(neighbors_reversed);
    free(scratch);
    free(marginals);
}



