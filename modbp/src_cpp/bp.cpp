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
    for (index_t i=0;i<n;++i)
    {
        printf("%lu ",arr[i]);
    }
    printf("\n");
}
void print_array(double *arr, index_t n)
{
    for (index_t i=0;i<n;++i)
    {
        printf("%f ",arr[i]);
    }
    printf("\n");
}



BP_Modularity::BP_Modularity(const vector<index_t>& _layer_membership, const vector<pair<index_t,index_t> > &intra_edgelist, const vector<pair<index_t,index_t> > &inter_edgelist, const index_t _n, const index_t _nt, const int _q, const double _beta, const double _omega, const double _resgamma, bool _verbose, bool _transform) :  layer_membership(_layer_membership), neighbor_count(_n), theta(_nt), num_edges(_nt), n(_n), nt(_nt), q(_q), beta(_beta), omega(_omega), resgamma(_resgamma), verbose(_verbose), transform(_transform), order(_n), rng((int)5)
{
    eps = 1e-8;
    computed_marginals = false;
    typedef pair<index_t, bool> ibpair;
    vector<vector< ibpair > > edges(n);
    uniform_real_distribution<double> pdist(0,1);
    uniform_int_distribution<index_t> destdist(0,n-1);
    neighbor_offset_map.resize(n);
    total_edges = 0;
    
    // TODO: go through all input edges and put them into the data structure along with categorization of their edge type
    
    
    for (auto p : intra_edgelist)
    {
        index_t i = p.first;
        index_t j = p.second;
        edges[i].push_back(ibpair(j,true));
        edges[j].push_back(ibpair(i,true));
        
        num_edges[layer_membership[i]] += 2;
        total_edges += 2;
    }
    
    for (auto p : inter_edgelist)
    {
        index_t i = p.first;
        index_t j = p.second;
        edges[i].push_back(ibpair(j,false));
        edges[j].push_back(ibpair(i,false));
        total_edges += 2;
    }
    
    beliefs.resize(q*total_edges);
    beliefs_old.resize(q*total_edges);
    beliefs_offsets.resize(n+1);
    
    neighbors.resize(total_edges);
    neighbors_reversed.resize(total_edges);
    neighbors_offsets.resize(n+1);
    neighbors_type.resize(total_edges);
    
    marginals.resize(q*n);
    marginals_old.resize(q*n);
    
    // set up offsets for fast access and copy graph structure into neighbors array
    size_t neighbors_offset_count = 0;
    size_t beliefs_offset_count = 0;
    int neighbor_c = 0;
    
    beliefs_offsets[0] = 0;
    neighbors_offsets[0] = 0;
    max_degree = 0;
    for (index_t i=0;i<n;++i)
    {
        beliefs_offset_count += q*edges[i].size();
        neighbors_offset_count += edges[i].size();
        neighbor_count[i] = (index_t) edges[i].size();
        
        beliefs_offsets[i+1] = beliefs_offset_count;
        neighbors_offsets[i+1] = neighbors_offset_count;
        
        max_degree = max(max_degree,(index_t) edges[i].size());
        
        for (index_t j=0;j<edges[i].size();++j)
        {
            //assert(neighbor_c < num_edges);
            neighbors_type[neighbor_c] = edges[i][j].second;
            neighbors[neighbor_c++] = edges[i][j].first;
            neighbor_offset_map[i][edges[i][j].first] = j;
        }
    }
    scratch.resize(max_degree*q);
    neighbor_c = 0;
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<edges[i].size();++j)
        {
            neighbors_reversed[neighbor_c++] = neighbor_offset_map[edges[i][j].first][i];
        }
    }
    
    reinit();
}

long BP_Modularity::run(unsigned long maxIters)
{
    
    change = 1;
    //unsigned long maxIters = 100;
    bool converged = false;
    for (unsigned long iter = 0; iter < maxIters; ++iter)
    {
        step();
        
        if (verbose)
            printf("Iteration %lu: change %f\n",iter+1,change);
        
        if (!changed)
        {
            converged = true;
            return iter;
            
            if (verbose)
                printf("Converged after %lu iterations.\n",iter+1);
        }
    }
    if (verbose)
        printf("Algorithm failed to converge after %lu iterations.\n",maxIters);
    return maxIters+1;
    
    
}

void BP_Modularity::compute_marginal(index_t i)
{
    const index_t nn = neighbor_count[i];
    index_t t = layer_membership[i];
    // iterate over all states
    double Z = 0;
    for (index_t s = 0; s < q; ++s)
    {
        marginals[q*i+s] = 0;
        for (index_t idx2 = 0; idx2<nn; ++idx2)
        {
            bool type = neighbors_type[neighbors_offsets[i]+idx2];
            double add;
            if (type==true)
            {
                // intralayer contribution
                add = log(1+scale*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
            }
            else
            {
                // interlayer contribution
                add = beta*omega*(beliefs[beliefs_offsets[i]+nn*s+idx2]);
            }
            marginals[q*i+s] += add;
        }
        // evaluate the rest of the update equation
        marginals[q*i+s] = exp(nn*theta[t][s] + marginals[q*i+s]);
        Z += marginals[q*i + s];
    }
    // normalize
    for (index_t s = 0; s < q; ++s)
    {
        marginals[q*i + s] /= Z;
    }
}

void BP_Modularity::step()
{
    changed = false;
    change = 0;
    
    // go through each node and update beliefs
    for (index_t node_idx = 0;node_idx<n;++node_idx)
    {
        index_t i = node_idx;
        index_t t = layer_membership[i];
        const index_t nn = neighbor_count[i];
        if (nn==0) continue;
        
        // first, see how much change we had to incoming beliefs so we know if we need to update
        double local_change = 0;
        for (index_t idx=beliefs_offsets[i];idx<beliefs_offsets[i+1];++idx)
        {
            local_change += fabs(beliefs[idx] - beliefs_old[idx]);
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
        
        // we should update the nodes contribution to theta
        compute_marginal(i);
        for (index_t s = 0; s < q; ++s)
        {
            theta[t][s] += nn * (marginals[q*i + s] - marginals_old[q*i + s]);
        }
        
        // update our record of what our incoming beliefs were for future comparison
        //memcpy(beliefs_old+beliefs_offsets[i], beliefs+beliefs_offsets[i], q*nn*sizeof(double));
        copy(beliefs.begin()+beliefs_offsets[i],beliefs.begin()+beliefs_offsets[i]+q*nn,beliefs_old.begin()+beliefs_offsets[i]);
        // do the same for marginals
        //memcpy(marginals_old + q*i, marginals + q * i, q * sizeof(double));
        copy(marginals.begin() + q*i, marginals.begin() + q*i + q, marginals_old.begin() + q*i);
        
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
                    bool type = neighbors_type[neighbors_offsets[i]+idx2];
                    double add;
                    if (type==true)
                    {
                        // intralayer contribution
                        add = log(1+scale*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
                    }
                    else
                    {
                        // interlayer contribution
                        //add = beta*omega*(beliefs[beliefs_offsets[i]+nn*s+idx2]);
                        add = 0;
                    }
                    scratch[nn*s+idx] += add;
                }
                // evaluate the rest of the update equation
                //scratch[nn*s+idx] = exp(prefactor*nn*theta[s] + scratch[nn*s+idx]);
                scratch[nn*s+idx] = exp(nn*theta[t][s] + scratch[nn*s+idx]);
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

void BP_Modularity::normalize(vector<double> & beliefs, index_t i)
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

void BP_Modularity::compute_marginals()
{
    for (index_t i = 0; i < n; ++i)
    {
        compute_marginal(i);
    }
}




double BP_Modularity::compute_bethe_free_energy()
{   //TODO
    // - 1/(b*beta) ( sum_i {log Z_i} - sum_{i,j \in E} log Z_{ij} + beta/4m \sum_i theta^2 )
    double bfe=0.0;
    return bfe;
}

double BP_Modularity::compute_factorized_free_energy()
{
    //TODO
    //Calculate the bethe free energy of the factorized state ( each node uniform on all communities)
    //log(1-1/q-exp(beta))
    double bffe=0.0;
    return bffe;
}


vector<vector<double> > BP_Modularity::return_marginals() { 
    // make sure the marginals are up-to-date
    compute_marginals();
    
    vector<vector<double> > ret(n);
    
    for (index_t i=0;i<n;++i)
    {
        ret[i].resize(q);
        for (index_t s=0;s<q;++s)
        {
            ret[i][s] = marginals[q*i+s];
        }
    }
    
    return ret;
}

void BP_Modularity::setBeta(double in, bool reset) {
    beta = in;
    
    reinit(reset,true);
}

void BP_Modularity::setResgamma(double in, bool reset) {
    resgamma = in;
    reinit(reset,true);
}

void BP_Modularity::setOmega(double in, bool reset) {
    omega = in;
    reinit(reset,true);
}

void BP_Modularity::setq(double new_q) {
    // rearrange the optimizer to have a different q and reinitialize
    q = new_q;

    
    beliefs.resize(q*total_edges);
    beliefs_old.clear();
    beliefs_old.resize(q*total_edges);
    marginals.resize(q*n);
    marginals_old.resize(q*n);
    scratch.resize(q*max_degree);
    
    theta.resize(q);
    
    // regenerate the beliefs_offsets
    index_t offset_count = 0;
    for (index_t i = 0; i<n;++i)
    {
        offset_count += q*neighbor_count[i];
        beliefs_offsets[i+1] = offset_count;
        if (!(offset_count <= q*total_edges))
        {
            printf("bad\n");
        }
    }
    
    reinit();

}

void BP_Modularity::reinit(bool init_beliefs,bool init_theta)
{
    scale = exp(beta)-1;
    if (init_beliefs)
        initializeBeliefs();
    if (init_theta)
        initializeTheta();
    copy(marginals.begin(),marginals.end(), marginals_old.begin());
}

void BP_Modularity::initializeBeliefs() { 
    // set starting value of beliefs
    // generate values for each state and then normalize
    normal_distribution<double> eps_dist(0,0.1);
    for (size_t idx = 0;idx<q*total_edges;++idx)
    {
        double val = eps_dist(rng);
        beliefs[idx] = truncate(1.0/q + val,q);
    }
    
    for (index_t i=0;i<n;++i)
    {
        normalize(beliefs,i);
    }
    
    // zero out old beliefs
    for (size_t i=0;i<q*total_edges;++i)
    {
        beliefs_old[i] = 0;
    }
}

void BP_Modularity::initializeTheta() { 
    // initialize values of theta for each layer
    for (index_t t = 0; t < nt; ++t)
    {
        // make sure the size is correct
        theta[t].resize(q);
        for (index_t s = 0; s<q;++s)
        {
            theta[t][s] = beta*resgamma*num_edges[t]/(q*num_edges[t]);
        }
    }
    for (index_t t = 0; t < nt; ++t)
    {
        compute_marginals();
        for (index_t i=0;i<n;++i)
        {
            index_t nn = n_neighbors(i);
            for (index_t s = 0; s<q;++s)
            {
                theta[t][s] += nn * marginals[q*i + s];
            }
        }
        // fold in prefactor to theta
        for (index_t s = 0; s<q;++s)
        {
            theta[t][s] *= beta*resgamma/num_edges[t];
        }
    }
}







