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



BP_Modularity::BP_Modularity(const index_t _n, const double p, const int _q, const double _beta, bool _simultaneous, bool _transform) : q(_q),n(_n), beta(_beta), simultaneous(_simultaneous), neighbor_count(_n), order(_n), rng(int(5)), transform(_transform)
{
    save = false;
    clock_t start = clock();
    
    printf("Constructing graph\n");
    
    scale = (1.0 - exp(-beta));
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
        }//*/
        /*
         index_t j = (i+1)%n;
         edges[i].push_back(j);
         edges[j].push_back(i);
         num_edges += 1;
         //*/
    }
    
    num_edges *= 2;
    
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
        //printf("%f\n",val);
        //beliefs[idx] = truncate(1.0/q + val,q);
        beliefs[idx] = unif_dist(rng);
    }//*/
    
    //print_beliefs(100);
    
    /*
    for (index_t i=0;i<n;++i)
    {
        const index_t nn = neighbor_count[i];
        for (int s=0;s<q;++s)
        {
            double val;
            if (s ==0) val = 0.9;
            else val = 0.05;
            for (index_t idx = 0;idx<nn;++idx)
            {
                beliefs[beliefs_offsets[i]+nn*s+idx] = val;
            }
        }
    }//*/
    for (index_t i=0;i<n;++i)
    {
        normalize(beliefs,i);
    }
    compute_marginals();
    
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
        stepNew();
        //changes.push_back(change);
        printf("Iteration %lu: change %f\n",iter+1,change);
        //print_beliefs(size_t(-1));
        // store the current marginals
        if (save)
        {
            compute_marginals();
        }
        
        
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

void BP_Modularity::stepNew()
{
    changed = false;
    change = 0;
    double * belief_write;
    if (simultaneous)
    {
        belief_write = beliefs_new;
    }
    else
    {
        belief_write = beliefs;
        //memcpy(beliefs_new, beliefs, q*num_edges*sizeof(double));
    }
    
    // go through each node
    //shuffle(order.begin(), order.end(), rng);
    uniform_int_distribution<index_t> choice_dist(0,n-1);
    for (index_t node_idx = 0;node_idx<n;++node_idx)
    {
        //index_t i = order[node_idx];
        //index_t i = choice_dist(rng);
        index_t i = node_idx;
        const index_t nn = neighbor_count[i];
        if (nn==0) continue;
        
        //*
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
        if (!simultaneous)
        {
            memcpy(beliefs_new+beliefs_offsets[i], beliefs+beliefs_offsets[i], q*nn*sizeof(double));
        }
        //*/
        
        // iterate over all states
        for (int s = 0; s < q;++s)
        {
            // incoming beliefs are already stored locally
            // go over each outgoing connection and figure out all of the outgoing beliefs, write to scratch space
            for (index_t idx=0; idx<nn; ++idx)
            {
                scratch[nn*s+idx] = 1;
                for (index_t idx2 = 0;idx2<nn;++idx2)
                {
                    if (idx2 == idx) continue;
                    //assert(beliefs_offsets[i]+nn*s + idx < q*num_edges);
                    double mult = (1.0-scale*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
                    //belief_write[beliefs_offsets[k]+nnk*s+idx_out] *= mult;
                    scratch[nn*s+idx]*=mult;
                }
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
                    scratch[nn*s+idx] = 0.5;
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
                
                belief_write[beliefs_offsets[k]+nnk*s+idx_out] = scratch[nn*s+idx];
            }
        }
    }
    // compute the total change
    /*
     change = 0;
     for (int i=0;i<q*num_edges;++i)
     {
     change += fabs(beliefs[i] - beliefs_new[i]);
     }
     change /= (2*q*num_edges);
     //*/
    
    if (simultaneous)
    {
        swap(beliefs,beliefs_new);
    }
    if (change > eps)
    {
        changed = true;
    }
}

void BP_Modularity::step()
{
    //printf("Entering step\n");
    //print_all_arrays();
    double * belief_write;
    if (simultaneous)
    {
        belief_write = beliefs_new;
    }
    else
    {
        memcpy(beliefs_new, beliefs, q*num_edges*sizeof(double));
        belief_write = beliefs;
    }
    
    // go through each node
    for (index_t i = 0;i<n;++i)
    {
        
        const index_t nn = neighbor_count[i];
        
        // iterate over all states
        for (int s = 0; s < q;++s)
        {
            // grab all incoming beliefs into scratch space
            for (index_t idx=0;idx<nn;++idx)
            {
                index_t k = neighbors[neighbors_offsets[i]+idx];
                // find which neighbor we are of k
                index_t idx2 = neighbors_reversed[neighbors_offsets[i]+idx];
                
                scratch[idx] = beliefs[beliefs_offsets[k]+n_neighbors(k)*s + idx2];
            }
            // go over each outgoing connection and update the belief
            for (index_t idx=0; idx<nn; ++idx)
            {
                belief_write[beliefs_offsets[i]+nn*s+idx] = 1;
                for (int idx2 = 0;idx2<nn;++idx2)
                {
                    if (idx2 == idx) continue;
                    //assert(beliefs_offsets[i]+nn*s + idx < q*num_edges);
                    double mult = (1.0-scale*scratch[idx2]);
                    belief_write[beliefs_offsets[i]+nn*s+idx] *= mult;
                }
            }
        }
        
        normalize(belief_write,i);
    }
    
    change = 0;
    // compute the total change
    for (int i=0;i<q*num_edges;++i)
    {
        change += fabs(beliefs[i] - beliefs_new[i]);
    }
    change /= (2*q*num_edges);
    if (simultaneous)
    {
        swap(beliefs,beliefs_new);
    }
}

// enforce normalization constraint on outgoing messsages of node i
// old version that used local storage of outgoing connections
/*void BP_Modularity::normalizeOld(double * beliefs, index_t i)
 {
 const index_t nn = neighbor_count[i];
 // iterate over all neighbors
 for (size_t idx2 = 0; idx2 < nn; ++ idx2)
 {
 // iterate over all states
 double sum = 1e-12;
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
 */

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

/*
 void BP_Modularity::print_all_arrays() {
 //print_array(beliefs,q*num_edges);
 print_array(beliefs_offsets,n+1);
 print_array(neighbors,num_edges);
 print_array(neighbors_offsets,n+1);
 }
 */

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
    int limit = 200;
    vector<byte> red(limit),green(limit),blue(limit);
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
        if ( q==3 && i < limit)
        {
            red[i] = 256*marginals[q*i];
            green[i] = 256*marginals[q*i+1];
            blue[i] = 256*marginals[q*i+2];
        }
    }
    if (q==3)
    {
        reds.push_back(red);
        greens.push_back(green);
        blues.push_back(blue);
    }
}

void data_save(const char* fn, vector<vector<byte> > data)
{
    FILE *fp = fopen(fn,"w+");
    for (int i=0;i<data.size();++i)
    {
        for (int j=0;j<data[i].size();++j)
        {
            fprintf(fp,"%d",data[i][j]);
            if (j < data[i].size()-1)
            {
                fprintf(fp,",");
            }
        }
        if (i<data.size()-1)
        {
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
}

void BP_Modularity::save_rgb()
{
    data_save("/Users/ben/Data/red.csv",reds);
    data_save("/Users/ben/Data/green.csv",greens);
    data_save("/Users/ben/Data/blue.csv",blues);
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

index_t BP_Modularity::return5(vector<pair<index_t, index_t> > edgelist) { 
    return 5;
}




