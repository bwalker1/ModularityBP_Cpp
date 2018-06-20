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
#include <algorithm>
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



BP_Modularity::BP_Modularity(const vector<index_t>& _layer_membership, const vector<pair<index_t,index_t> > &intra_edgelist, const vector<pair<index_t,index_t> > &inter_edgelist, const index_t _n, const index_t _nt, const int _q, const double _beta, const double _omega, const double _resgamma, bool _verbose, bool _transform) :  layer_membership(_layer_membership), neighbor_count(_n), theta(_nt), num_edges(_nt), n(_n), nt(_nt), q(_q), beta(_beta), omega(_omega), resgamma(_resgamma), verbose(_verbose), transform(_transform), order(_n), rng(time(NULL))
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
        order[i] = i;
        
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
    
    compute_bfe = false;
}

long BP_Modularity::run(unsigned long maxIters)
{
    
    change = 1;
    //unsigned long maxIters = 100;
    bool converged = false;
    iter = 0;
    for (iter = 0; iter < maxIters; ++iter)
    {
        step();
        
        // monitor changes
        
        
        if (verbose)
            printf("Iteration %lu: change %f\n",iter+1,change);
        
        if (!changed)
        {
            converged = true;
            
            if (verbose)
                printf("Converged after %lu iterations.\n",iter+1);
            
            return iter;
        }
    }
    if (verbose)
        printf("Algorithm failed to converge after %lu iterations.\n",maxIters);
    return maxIters+1;
    
    
}

void BP_Modularity::compute_marginal(index_t i, bool do_bfe_contribution)
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
                add = log(1+scaleOmega*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
            }
            marginals[q*i+s] += add;
        }
        // evaluate the rest of the update equation
        marginals[q*i+s] = exp(nn*theta[t][s] + marginals[q*i+s]);
        
        Z += marginals[q*i + s];
        
        assert(Z > 0);
        assert(!isnan(Z));
        
        if (!(Z > 0 && !isnan(Z)))
        {
            //printf("Z is not correct\n");
        }
    }
    if (do_bfe_contribution)
    {
        bfe += log(Z);
    }
    // normalize
    for (index_t s = 0; s < q; ++s)
    {
        if (Z > 0)
        {
            marginals[q*i + s] /= Z;
        }
        else
        {
            marginals[q*i + s] = 1.0/q;
        }
        assert(!isnan(marginals[q*i + s]));
    }
}

void BP_Modularity::step()
{
    changed = false;
    change = 0;
    
    bool fast_convergence = false;
    
    if (compute_bfe)
    {
        bfe = 0.0;
    }
    
    // shuffle order
    std::shuffle(order.begin(),order.end(),rng);
    // go through each node and update beliefs
    for (index_t node_idx = 0;node_idx<n;++node_idx)
    {
        index_t i;
        if (iter%2 == 0)
        {
            i = order[node_idx];
        }
        else
        {
            i = node_idx;
        }
        
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
        if (fast_convergence)
        {
            if (local_change < eps)
            {
                // not enough change in incoming beliefs to warrant an update
                //continue;
            }
        }
        // if we changed any nodes, set this to true so we know we haven't converged
        //changed = true;
        change += local_change;
        
        // we should update the nodes contribution to theta
        compute_marginal(i);
        for (index_t s = 0; s < q; ++s)
        {
            theta[t][s] += -beta*resgamma/(num_edges[t])* nn * (marginals[q*i + s] - marginals_old[q*i + s]);
        }
        
        // update our record of what our incoming beliefs were for future comparison
        copy(beliefs.begin()+beliefs_offsets[i],beliefs.begin()+beliefs_offsets[i]+q*nn,beliefs_old.begin()+beliefs_offsets[i]);
        // do the same for marginals
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
                        add = log(1+scaleOmega*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
                    }
                    scratch[nn*s+idx] += add;
                }
                // evaluate the rest of the update equation
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
            if (compute_bfe)
            {
                bfe -= log(sum);
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
                assert(!isnan(scratch[nn*s+idx]));
                beliefs[beliefs_offsets[k]+nnk*s+idx_out] = scratch[nn*s+idx];
            }
        }
    }
    if (compute_bfe)
    {
        compute_marginals(true);
        
        for (index_t t=0;t<nt;++t)
        {
            double temp = 0;
            for (index_t s=0;s<q;++s)
            {
                double temp2 = theta[t][s];
                temp += temp2*temp2;
            }
            bfe += beta/(2*num_edges[t]) * temp;
        }
        bfe /= (beta*n);
    }
    
    if (!fast_convergence)
    {
        if (change > eps)
        {
            changed = true;
        }
        else
        {
            changed = false;
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
            assert(sum==1);
            if (sum > 0)
            {
                beliefs[beliefs_offsets[i]+nn*s+idx2] /= sum;
            }
            else
            {
                beliefs[beliefs_offsets[i]+nn*s+idx2] = 1.0/q;
            }
            assert(!isnan(beliefs[beliefs_offsets[i]+nn*s+idx2] = 1.0/q));
        }
    }
}

void BP_Modularity::compute_marginals()
{
    compute_marginals(false);
}

void BP_Modularity::compute_marginals(bool do_bfe_contribution)
{
    for (index_t i = 0; i < n; ++i)
    {
        compute_marginal(i,do_bfe_contribution);
    }
}




double BP_Modularity::compute_bethe_free_energy()
{
    // - 1/(n*beta) ( sum_i {log Z_i} - sum_{i,j \in E} log Z_{ij} + beta/4m \sum_s theta_s^2 )
    if (compute_bfe == false)
    {
        compute_bfe = true;
        step();
        compute_bfe = false;
    }
    return bfe;
}

double BP_Modularity::compute_factorized_free_energy()
{
    //Calculate the bethe free energy of the factorized state ( each node uniform on all communities)
    //log(1-1/q-exp(beta))
    double bffe=log(1-1.0/q - exp(beta));
    return bffe;
}


vector<vector<double>> BP_Modularity::return_marginals() {
    // make sure the marginals are up-to-date
    compute_marginals();
    
    vector<vector<double>> ret(n);
    
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
    
    // regenerate the beliefs_offsets
    index_t offset_count = 0;
    for (index_t i = 0; i<n;++i)
    {
        offset_count += q*neighbor_count[i];
        beliefs_offsets[i+1] = offset_count;
    }
    
    reinit();
    
}

void BP_Modularity::reinit(bool init_beliefs,bool init_theta)
{
    if (beta==0)
    {
        beta = compute_bstar(omega,q);
    }
    scale = exp(beta)-1;
    scaleOmega = exp(beta*omega)-1;
    if (init_beliefs)
        initializeBeliefs();
    if (init_theta)
        initializeTheta();
    copy(marginals.begin(),marginals.end(), marginals_old.begin());
}



//void shuffleBeliefs(vector<vector<double>> in_beliefs){
//    //rearrange each of outgoing beliefs for each node
//    //according to permutation vector
//    //input is a n by q vector
//}

void BP_Modularity::initializeBeliefs() { 
    // set starting value of beliefs
    // generate values for each state and then normalize
    normal_distribution<double> eps_dist(0,0.1);
    /*
    for (index_t idx=0;idx<n;++idx)
    {
        const index_t nn = neighbor_count[idx];
        
        bool group1 =((idx-(n/nt)*layer_membership[idx])) < (n*1.0/(2.0*nt));
        
        for (size_t s = 0; s < q; ++s)
        {
            for (index_t idx2 = 0; idx2 < nn; ++ idx2)
            {
                index_t k = neighbors[neighbors_offsets[idx]+idx2];
                const index_t nnk = neighbor_count[k];
                index_t idx_out = neighbors_reversed[neighbors_offsets[idx]+idx2];
                beliefs[beliefs_offsets[idx]+nn*s+idx2] = int(group1);
                //printf("%f\n",beliefs[beliefs_offsets[k]+nnk*s+idx_out]);
            }
        }
    }
    */
    //compute_marginals();
    for (index_t i=0;i<n;++i)
    {
        //printf("%f\n",marginals[i]);
    }
    
    for (index_t i=0;i<beliefs.size();++i)
    {
        beliefs[i] = truncate(1.0/q + eps_dist(rng),q);
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
    theta.resize(nt);
    for (index_t t = 0; t < nt; ++t)
    {
        // make sure the size is correct
        theta[t].resize(q);
        for (index_t s = 0; s<q;++s)
        {
            theta[t][s] = beta*resgamma/(q);
            //theta[t][s] = 0;
        }
    }
    
    compute_marginals();
    for (index_t t = 0; t < nt; ++t)
    {
        for (index_t s=0;s<q;++s)
        {
            theta[t][s]=0;
        }
    }
    for (index_t i=0;i<n;++i)
    {
        index_t t = layer_membership[i];
        index_t nn = neighbor_count[i];
        for (index_t s = 0; s<q;++s)
        {
            theta[t][s] += nn * marginals[q*i + s];
        }
    }
    for (index_t t = 0; t < nt; ++t)
    {
        // fold in prefactor to theta
        for (index_t s = 0; s<q;++s)
        {
            theta[t][s] *= -(beta*resgamma/num_edges[t]);
        }
    }
}

double s(double beta, double omega, double q, double c)
{
    double eb = exp(beta);
    double ewb = exp(omega*beta);
    
    double temp1 = ((eb-1)/(eb-1+q));
    double temp2 = ((ewb-1)/(ewb-1+q));
    return c*temp1*temp1 + 2*temp2*temp2;
}

double sp(double beta, double omega, double q, double c)
{
    double eb = exp(beta);
    double ewb = exp(omega*beta);
    
    double temp1 = eb - 1 + q;
    double temp2 = ewb- 1 + q;
    return 2*c*eb*(eb-1)*q/(temp1*temp1*temp1) + 4*ewb * (ewb-1)*q*omega/(temp2*temp2*temp2);
}

void BP_Modularity::merge_communities(vector<index_t> merges)
{
    // figure out the new number of communities
    index_t q_new = *max_element(merges.begin(),merges.end());
    index_t q_old = q;
    vector<double> beliefs_temp(beliefs);
    vector<index_t> beliefs_offsets_temp(beliefs_offsets);
    setq(q_new);
    
    // zero out beliefs
    for (index_t i=0;i<beliefs.size();++i)
    {
        beliefs[i] = 0;
    }
    
    // write in correct values for beliefs
    for (index_t i=0;i<n;++i)
    {
        index_t nn = neighbor_count[i];
        for (int s=0;s<q_old;++s)
        {
            for (index_t idx2=0;idx2<nn;++idx2)
            {
                beliefs[beliefs_offsets[i]+q_new*i+merges[s]] += beliefs_temp[beliefs_offsets_temp[i]+nn*s+idx2];
            }
        }
    }
    
    
}

void BP_Modularity::permute_beliefs(vector<vector<index_t> > permutation)
{
    //
    // go through each layer and apply the permutation described to it
    if (permutation.size() != nt)
    {
        fprintf(stderr,("Permutation vector list has wrong length %d != %d \n"),permutation.size(),nt);
        return;
    }
    vector<double> vals(q); //storage for current beliefs
    for (index_t i = 0; i < nt; ++i)
    {

        index_t nn = neighbor_count[i];
        for (index_t idx2=0; idx2<nn ;++idx2)
        {
            //copy beliefs into temp based on permutation order
            //i.e. the new kth belief is given by kth value of permutation vector
            for (int k=0;k<permutation[i].size();++k)
            {
                vals[k] = beliefs[beliefs_offsets[i]+nn*permutation[i][k]+idx2];
            }
            //go back through and copy back over onto beliefs reordered
            for (int k=0;k<permutation[i].size();++k)
            {
                beliefs[beliefs_offsets[i]+nn*k+idx2]=vals[k];
            }
        }
    }
   //maybe i'm missing something here but vals is a vector of length q?
   //wouldn't this just set all of the beliefs to be the same for each node?
   //see above swap
    // write out the beliefs
//    for (index_t i = 0; i < nt; ++i)
//    {
//        index_t nn = neighbor_count[i];
//        for (index_t idx2=0;idx2<nn;++idx2)
//        {
//            for (int s=0;s<q;++s)
//            {
//                beliefs[beliefs_offsets[i]+nn*s+idx2] = vals[s];
//            }
//        }
//    }

}

double BP_Modularity::compute_bstar(double omega_in,int q_in)
{
    // currently this assumes multiplex graph
    
    // compute c - decide on the right way
    // the simple average degree
    double c = accumulate(num_edges.begin(), num_edges.end(), 0.0)/n;
    
    // compute the excess degree
    double d2 = 0;
    double d = 0;
    for (int i=0;i<n;++i)
    {
        double nn = neighbor_count[i];
        if (layer_membership[i] == 0)
        {
            nn -= 1;
        }
        if (layer_membership[i] == nt-1)
        {
            nn -= 1;
        }
        
        d2 += nn*nn;
        d += nn;
    }
    c = d2/d - 1;
    
    // bisection/newton hybrid method
    double xl=0, xr=1;
    double xn;
    
    // find bounding interval
    while (s(xr,omega_in,q_in,c) < 1)
    {
        xr *= 2;
    }
    
    // start newton's from midpoint
    xn = (xl+xr)/2;
    double yn = s(xn,omega_in,q_in,c);
    double ypn = sp(xn,omega_in,q_in,c);
    
    int maxiters = 100;
    for (int iters=0;iters<maxiters;)
    {
        // try a newton step
        
        xn -= (yn - 1)/ypn;
        yn = s(xn,omega_in,q_in,c);
        ypn = sp(xn,omega_in,q_in,c);
        
        // check if this is in our bounding interval
        if (xl < xn && xn < xr)
        {
            // narrow our interval using newton point
            if (yn > 1)
            {
                xr = xn;
            }
            else
            {
                xl = xn;
            }
        }
        else
        {
            // narrow our interval using bisection
            double xc = (xl + xr)/2;
            if (s(xc,omega_in,q_in,c)>1)
            {
                xr = xc;
            }
            else
            {
                xl = xc;
            }
            
            // restart newton's at the new midpoint
            xn = (xl + xr)/2;
            yn = s(xn,omega_in,q_in,c);
            ypn = sp(xn,omega_in,q_in,c);
        }
        
        // check for convergence
        if (xr - xl < 1e-6)
        {
            return (xl + xr)/2;
        }
    }
    
    return (xl+xr)/2;
}
