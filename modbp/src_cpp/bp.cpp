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
        fprintf(stdout,"%f ",arr[i]);
    }
    fprintf(stdout,"\n");
}

void BP_Modularity::initialize_edge_data(const vector<pair<index_t,index_t> > &intra_edgelist, const vector<pair<double,double>> &intra_edgeweight,const vector<double> &inter_edgeweight, const vector<pair<index_t,index_t> > &inter_edgelist, vector<vector< edge_data > > &edges)
{
    for (index_t k=0;k<intra_edgelist.size();++k)
    {
        auto p = intra_edgelist[k];
        double w;
        index_t l;
        
        //l gives the layer that the intralayer edge originally came from
        l = (index_t) intra_edgeweight[k].first;//typecast to index
        w = intra_edgeweight[k].second;
        
        index_t i = p.first;
        index_t j = p.second;
        edges[i].push_back(edge_data(j,l,true,w));
        edges[j].push_back(edge_data(i,l,true,w));
        
        num_edges[l] += w;
        //        num_strength[layer_membership[i]] += w;
        
        total_strength += w;
        total_edges += 2;
        if (i!=j) { total_belief_edges+=2;}
    }
    
    for (index_t k=0;k<inter_edgelist.size();k++)
    {
        
        auto p = inter_edgelist[k];
        double w;
        
        w = inter_edgeweight[k];
        
        index_t i = p.first;
        index_t j = p.second;
        //We don't store layer information on the interlayer edges
        //We do store weight however
        edges[i].push_back(edge_data(j,-1,false,w));
        edges[j].push_back(edge_data(i,-1,false,w));
        neighbor_count_interlayer[i]+=1;
        neighbor_count_interlayer[j]+=1;
        
        total_edges += 2;
        total_strength += w;
        
        if (i!=j) { total_belief_edges+=2;}
    }
}

BP_Modularity::BP_Modularity(const vector<vector<index_t>>& _layer_membership, const vector<pair<index_t,index_t> > &intra_edgelist, const vector<pair<double,double>> &intra_edgeweight,const vector<double> &inter_edgeweight, const vector<pair<index_t,index_t> > &inter_edgelist, const index_t _n, const index_t _nlayers,  const int _q, const double _beta, const double _omega, const double _dumping_rate, const double _resgamma, bool _verbose, bool _transform, bool _parallel) :  layer_membership(_layer_membership), neighbor_count(_n), neighbor_count_interlayer(_n), node_strengths(_n), theta(_nlayers), num_edges(_nlayers), n(_n), nlayers(_nlayers), q(_q), beta(_beta), omega(_omega), dumping_rate(_dumping_rate), resgamma(_resgamma), verbose(_verbose), transform(_transform), parallel(_parallel), order(_n), rng(time(NULL))
{
    //TODO consider only allowing weighted option
    if (intra_edgeweight.size() > 0)
    {
        weighted = true;
    }
    else
    {
        weighted = false;
    }
    
    //fprintf(stdout,"is_bipartite:%s\n", is_bipartite ? "true" : "false");
    //fprintf(stderr,"Constructing %s graph: length %d weights\n",weighted?"weighted":"unweighted",intra_edgeweight.size());
    eps = 1e-8;
    computed_marginals = false;
    typedef pair<index_t, bool> ibpair;
    vector<vector< edge_data > > edges(n);
    uniform_real_distribution<double> pdist(0,1);
    uniform_int_distribution<index_t> destdist(0,n-1);
    neighbor_offset_map.resize(n);
    total_edges = 0;
    total_belief_edges=0;
    total_strength=0;

    // this initializes the edges variable (by reference)
    initialize_edge_data(intra_edgelist, intra_edgeweight, inter_edgeweight, inter_edgelist, edges);

    //fprintf(stdout,"total_belief_edges: %d\n",total_belief_edges);
    beliefs.resize(q*total_belief_edges);
    beliefs_old.resize(q*total_belief_edges);
    beliefs_offsets.resize(n+1);
    
    neighbors.resize(total_belief_edges);
    neighbors_reversed.resize(total_belief_edges);
    neighbors_offsets.resize(n+1);
    neighbors_type.resize(total_belief_edges);
    
    edge_weights.resize(total_belief_edges);
    scaleEdges.resize(total_belief_edges);
    
    marginals.resize(q*n);
    marginals_old.resize(q*n);
    
    // set up offsets for fast access and copy graph structure into neighbors array
    size_t neighbors_offset_count = 0;
    size_t beliefs_offset_count = 0;
    int neighbor_c = 0;
    double c_strength=0; // strength of current node
    beliefs_offsets[0] = 0;
    neighbors_offsets[0] = 0;
    max_degree = 0;
    for (index_t i=0;i<n;++i)
    {
        order[i] = i;
        //count number of self loops.  we don't pass
        //beliefs along these self loops though they do contribute to field/null term
        int nself_loops=0;
        for (index_t j=0;j<edges[i].size();++j){
            if (edges[i][j].target==i){
                nself_loops+=1;
            }
        }
        int num_belief_edges=edges[i].size()-nself_loops;
        
        beliefs_offset_count += q*(num_belief_edges);
        neighbors_offset_count += num_belief_edges;
        neighbor_count[i] = (index_t) num_belief_edges;
        //fprintf(stdout,"neighborcount[%d]: %d\n",i,neighbor_count[i]);
        
        beliefs_offsets[i+1] = beliefs_offset_count;
        neighbors_offsets[i+1] = neighbors_offset_count;
        
        //        max_degree = max(max_degree,(index_t) edges[i].size());
        max_degree = max(max_degree,(index_t) num_belief_edges);
        
        c_strength=0;
        node_strengths[i].resize(nlayers);
        for (index_t lay=0;lay<node_strengths[i].size();lay++ ){
            node_strengths[i][lay]=0;
        }
        
        for (index_t j=0;j<edges[i].size();++j)
        {
            //            fprintf(stdout,"(%d,%d):%f\n",i,edges[i][j].target,edges[i][j].weight);
            //assert(neighbor_c < num_edges);
            index_t clayer = edges[i][j].layer;
            if (edges[i][j].type){ //only add in the intralyer strength (including self loops)
                node_strengths[i][clayer]+=edges[i][j].weight;
            }
            
            if (edges[i][j].target!=i){
                //we only pass beliefs if it's not a self loop
                if (edges[i][j].type)
                {edge_weights[neighbor_c] = edges[i][j].weight;}
                else //interlayer edges are multiplied by omega
                {edge_weights[neighbor_c] = omega*edges[i][j].weight;}
                
                neighbors_type[neighbor_c] = edges[i][j].type;
                neighbors[neighbor_c] = edges[i][j].target;
                neighbor_offset_map[i][edges[i][j].target] = j;
                neighbor_c++;
            }
        }
    }

    scratch.resize(max_degree*q);
    neighbor_c = 0;
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<edges[i].size();++j)
        {
            if (edges[i][j].target!=i)
            {
                neighbors_reversed[neighbor_c++] = neighbor_offset_map[edges[i][j].target][i];
            }
        }
    }
    reinit();
    
    compute_bfe = false;
}

vector<double> BP_Modularity::run(unsigned long maxIters)
{
    
    change = 1;
    vector<double> changes(0);
    //unsigned long maxIters = 100;
    bool converged = false;
    iter = 0;
    for (iter = 0; iter < maxIters; ++iter)
    {
        step();
        
        // monitor changes
        if (verbose)
            printf("Iteration %lu: change %f\n",iter+1,change);
        
        changes.push_back(change);
        
        if (!changed)
        {
            converged = true;
            if (verbose)
                printf("Converged after %lu iterations.\n",iter+1);
            //            return iter;
            return changes;
            
        }
    }
    if (verbose)
        printf("Algorithm failed to converge after %lu iterations.\n",maxIters);
    //    return maxIters+1;
    return changes;
    
}

void BP_Modularity::compute_marginal(index_t i, bool do_bfe_contribution)
{
    const index_t nn = neighbor_count[i];
    vector<double> c_strength = node_strengths[i];
    
    vector<index_t> cur_layers = layer_membership[i];
    // iterate over all states
    double Z = 0;
    for (index_t s = 0; s < q; ++s)
    {
        //marginals[q*i+s] = 0;
        double mul = 1.0;
        
        for (index_t idx2 = 0; idx2<nn; ++idx2)
        {
            mul *= (1+scaleEdges[neighbors_offsets[i]+idx2]*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
        }
        // evaluate the rest of the update equation
        double field=0;

        for (index_t lay=0;lay<nlayers;lay++)
        {
            field += c_strength[lay]*theta[lay][s];
        }
        
        //double temp_inside = field + marginals[q*i+s];
        //marginals[q*i+s] = exp(temp_inside);
        mul *= exp(field);
        marginals[q*i+s] = mul;
        
        Z += marginals[q*i + s];
        
        assert(Z > 0);
        assert(!::isnan(Z));
    }
    if (do_bfe_contribution)
    {
        bfe += log(Z);
    }
    // normalize
    if (!std::isinf(Z))
    {
        for (index_t s = 0; s < q; ++s)
        {
            double temp = marginals[q*i + s];
            if (Z > 0)
            {
                marginals[q*i + s] /= Z;
            }
            else
            {
                marginals[q*i + s] = 1.0/q;
            }
        }
    }
    else
    {
        // count how many large marginals there are
        int inf_count = 0;
        for (index_t s = 0; s < q; ++s)
        {
            if (std::isinf(marginals[q*i + s]))
            {
                ++inf_count;
            }
        }
        for (index_t s = 0; s < q; ++s)
        {
            if (std::isinf(marginals[q*i + s]))
            {
                marginals[q*i+s] = 1.0/(inf_count);
            }
            else
            {
                marginals[q*i+s] = 0;
            }
        }
    }
}

// Compute the contribution along the edge from idx -> i, state s, using beliefs from beliefs_p
inline double BP_Modularity::edge_equation(vector<double> *beliefs_p, index_t i, index_t s, index_t nn, index_t idx)
{
    return (1 + scaleEdges[neighbors_offsets[i] + idx] * ((*beliefs_p)[beliefs_offsets[i] + nn * s + idx]));
}

bool BP_Modularity::step()
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

        i = order[node_idx];
        
        
        vector<index_t> cur_layer = layer_membership[i];
        const index_t nn = neighbor_count[i];
        vector<double> c_strength = node_strengths[i];
        
        if (nn==0) continue;
        
        // first, see how much change we had to incoming beliefs so we know if we need to update
        double local_change = 0;
        for (index_t idx=beliefs_offsets[i];idx<beliefs_offsets[i+1];++idx)
        {
            local_change += fabs(beliefs[idx] - beliefs_old[idx]);
        }
        local_change /= q*nn;

        // if we changed any nodes, set this to true so we know we haven't converged
        //changed = true;
        change += local_change;

		// depending on if we're doing serial or parallel updates, we read beliefs from either the beliefs or beliefs_old variable
		vector<double> * beliefs_to_use_p;
		if (parallel)
		{
			// read out of old beliefs so we aren't changing them as we go node to node
			beliefs_to_use_p = &beliefs_old;
		}
		else
		{
			// make sure we use the most up-to-date (if we already changed some this step)
			beliefs_to_use_p = &beliefs;
		}
        
        // we should update the nodes contribution to theta
        compute_marginal(i);
        for (index_t s = 0; s < q; ++s)
        {
            for(index_t lay=0;lay<nlayers;lay++)
            {
                //each node contributes to each layer's theta according to edges it has in that layer
                theta[lay][s] += -beta*resgamma/(2*num_edges[lay])* c_strength[lay] * (marginals[q*i + s] - marginals_old[q*i + s]);
            }
        }
        
        // update our record of what our incoming beliefs were for future comparison
        copy(beliefs.begin()+beliefs_offsets[i],beliefs.begin()+beliefs_offsets[i]+q*nn,beliefs_old.begin()+beliefs_offsets[i]);
        // do the same for marginals
        copy(marginals.begin() + q*i, marginals.begin() + q*i + q, marginals_old.begin() + q*i);
        
        // iterate over all states
        for (int s = 0; s < q;++s)
        {
            // incoming beliefs are already stored locally
            
            // precompute the belief stuff so that we aren't wasting so much time (hopefully)
            double total_factor = 1.0;
            for (index_t idx=0; idx<nn; ++idx)
            {
				total_factor *= (1 + scaleEdges[neighbors_offsets[i] + idx] * ((*beliefs_to_use_p)[beliefs_offsets[i] + nn * s + idx]));
            }
            
            // figure out the sum of logs part of the update equation that uses the incoming beliefs
            for (index_t idx=0; idx<nn; ++idx)
            {
                // put in the total factor minus the one we aren't factoring in
				scratch[nn*s + idx] = total_factor / (1 + scaleEdges[neighbors_offsets[i] + idx] * ((*beliefs_to_use_p)[beliefs_offsets[i] + nn * s + idx]));
                // evaluate the rest of the update equation
                
                double field=0;

                for(index_t lay=0;lay<nlayers;lay++)
                {
                    field += c_strength[lay]*theta[lay][s];
                }

                //double temp = exp(field + scratch[nn*s+idx]);
                //scratch[nn*s+idx] = temp;
                scratch[nn*s+idx] *= exp(field);
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
                //                printf("in loop sum %.3f\n",sum);
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
                assert(!::isnan(scratch[nn*s+idx]));
                beliefs[beliefs_offsets[k]+nnk*s+idx_out] = dumping_rate*scratch[nn*s+idx] +(1.0-dumping_rate)*beliefs[beliefs_offsets[k]+nnk*s+idx_out]; //weighted average of previous and current belief
            }
        }
    }
    
    if (compute_bfe)
    {
        compute_marginals(true);
        
        // compute pair terms of bethe free energy
        // note: this double counts edges so we have to divide by 2
        for (index_t node_idx = 0;node_idx<n;++node_idx)
        {
            index_t i = node_idx;     // the id of the source node
            // iterate over neighbors
            const index_t nn = neighbor_count[i];
            for (index_t idx=0; idx<nn; ++idx)
            {
                index_t k = neighbors[neighbors_offsets[i]+idx];    // the id of the target node
                const index_t nnk = neighbor_count[k];
                index_t idx_out = neighbors_reversed[neighbors_offsets[i]+idx];
                double sum = 0;
                
                // figure out our e^something
                double scaleHere = scaleEdges[neighbors_offsets[i]+idx];

                // iterate over all states of first node
                for (int s = 0; s < q;++s)
                {
                    // iterate over all states of second node
                    for (int t = 0; t<q; ++t)
                    {
                        // belief from source to target
                        double psi1 = beliefs[beliefs_offsets[i]+nn*s+idx];
                        // belief from target to source
                        double psi2 = beliefs[beliefs_offsets[k]+nnk*s+idx_out];
                        // ternary operator for delta_st (Kronecker delta function)
                        
                        sum += (s==t?(scaleHere+1):1)*psi1*psi2;
                        
                    }
                }
                // add contribution to bfe and divide by 2 to avoid double counting
                bfe -= log(sum)/2;
            }
        }
        
        //contribution of non-edges (i.e. from the null model)
        for (index_t lay=0;lay<nlayers;++lay)
        {
            double temp = 0;
            for (index_t s=0;s<q;++s)
            {
                double temp2 = theta[lay][s];
                temp += temp2*temp2;
            }
            bfe += beta/(4*num_edges[lay]) * temp;
        }
        bfe /= (-1*beta*n); //normalize out by beta a n
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
    return changed;
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
            //assert(sum==1);  // TODO: why was this  here?
            if (sum > 0)
            {
                beliefs[beliefs_offsets[i]+nn*s+idx2] /= sum;
            }
            else
            {
                beliefs[beliefs_offsets[i]+nn*s+idx2] = 1.0/q;
            }
            assert(!::isnan(beliefs[beliefs_offsets[i]+nn*s+idx2] = 1.0/q));
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

vector <double> BP_Modularity::getBeliefs(){
    vector<double> outbeliefs(beliefs.size());
    for (index_t i =0 ;i<beliefs.size(); ++i ){
        outbeliefs[i]=beliefs[i];
    }
    return outbeliefs;
}

void BP_Modularity::setBeta(double in, bool reset) {
    beta = in;
    reinit(reset,reset);
}

void BP_Modularity::setResgamma(double in, bool reset) {
    resgamma = in;
    reinit(reset,true);
}

void BP_Modularity::setOmega(double in, bool reset) {
    omega = in;
    reinit(reset,reset);
}

void BP_Modularity::setDumpingRate(double in) {
    dumping_rate = in;
}

void BP_Modularity::setq(double new_q) {
    // rearrange the optimizer to have a different q and reinitialize
    q = new_q;
    
    
    beliefs.resize(q*total_belief_edges);
    beliefs_old.clear();
    beliefs_old.resize(q*total_belief_edges);
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
        beta=eps; //must be non-zero.  make very small
    }
    
    
    for (index_t i=0;i<total_belief_edges;++i)
    {
        //omega has already been baked into edge weights for interlayer
        scaleEdges[i] = exp(beta*edge_weights[i])-1;
    }
    
    
    
    if (init_beliefs)
        initializeBeliefs();
    
    if (init_theta)
    {
        initializeTheta();
    }
    
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
    
    
    for (index_t i=0;i<beliefs.size();++i)
    {
        beliefs[i] = truncate(1.0/q + eps_dist(rng),q);
    }
    
    for (index_t i=0;i<n;++i)
    {
        normalize(beliefs,i);
    }
    
    // zero out old beliefs
    for (size_t i=0;i<q*total_belief_edges;++i)
    {
        beliefs_old[i] = 0;
    }
}

void BP_Modularity::initializeTheta() { 
    // initialize values of theta for each layer
    theta.resize(nlayers);
    for (index_t lay = 0; lay <nlayers; ++lay)
    {
        // make sure the size is correct
        theta[lay].resize(q);
        for (index_t s = 0; s<q;++s)
        {
            theta[lay][s] = beta*resgamma/(q);
            //theta[t][s] = 0;
            
        }
    }
    
    compute_marginals();
    
    for (index_t lay = 0; lay <nlayers; ++lay)
    {
        for (index_t s=0;s<q;++s)
        {
            theta[lay][s]=0;
            
        }
    }
    for (index_t i=0;i<n;++i)
    {
        vector<double> cur_strength = node_strengths[i];
        //        index_t nn = neighbor_count[i];
        
        for (index_t s = 0; s<q;++s)
        {
            for (index_t lay=0;lay<nlayers;lay++)
            {
                theta[lay][s] += cur_strength[lay] * marginals[q*i + s];
            }
            
        }
        
    }
    for (index_t lay = 0; lay <nlayers; ++lay)
    {
        // fold in prefactor to theta
        for (index_t s = 0; s<q;++s)
        {
            theta[lay][s] *= -(beta*resgamma/(2*num_edges[lay]));
        }
    }
}

void BP_Modularity::merge_communities(vector<index_t> merges)
{
    // figure out the new number of communities
    index_t q_new = *max_element(merges.begin(),merges.end()) + 1;
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
                index_t idx_1 = beliefs_offsets[i]+nn*merges[s]+idx2;
                index_t idx_2 = beliefs_offsets_temp[i]+nn*s+idx2;
                
                beliefs[idx_1] += beliefs_temp[idx_2];
            }
        }
    }
    
    
}

void BP_Modularity::permute_beliefs(vector<vector<index_t> > permutation)
{
    //
    // go through each layer and apply the permutation described to it
    if (permutation.size() !=nlayers)
    {
        fprintf(stderr,("Permutation vector list has wrong length %d != %d \n"),permutation.size(),nlayers);
        return;
    }
    vector<double> vals(q); //storage for current beliefs
    vector<index_t> c_layer_ind;
    
    for (index_t i = 0; i < n; ++i) //iterate through all nodes (n)
    {
        c_layer_ind = layer_membership[i];
        index_t nonzero;
        bool found= false;
        for (index_t li=0;li<nlayers;li++){
            if (c_layer_ind[li]!=0){
                
                if (found == false)
                {
                    nonzero=li;
                    found=true;
                }
                else{
                    throw "Permuting beliefs only works when each node is in a single layer.  Found multiple layers for node i";
                }
                
            }
        }
        index_t nn = neighbor_count[i];
        
        for (index_t idx2=0; idx2<nn ;++idx2)
        {
            //copy beliefs into temp based on permutation order
            //i.e. the new kth belief is given by kth value of permutation vector
            for (int k=0;k<permutation[nonzero].size();++k)
            {
                vals[k] = beliefs[beliefs_offsets[i]+nn*permutation[nonzero][k]+idx2];
            }
            //go back through and copy back over onto beliefs reordered
            for (int k=0;k<permutation[nonzero].size();++k)
            {
                beliefs[beliefs_offsets[i]+nn*k+idx2]=vals[k];
            }
        }
    }
}

void BP_Modularity::setBeliefs(vector<double> new_beliefs )
{
    //
    // go through each layer and apply the permutation described to it
    if (new_beliefs.size() != beliefs.size())
    {
        fprintf(stderr,("New beliefs must be same size as old %d != %d \n"),beliefs.size(),new_beliefs.size());
        return;
    }
    vector<double> vals(q); //storage for current beliefs
    size_t c_layer_ind;
    for (index_t i = 0; i < beliefs.size(); ++i) //iterate through all nodes (n)
    {
        beliefs[i] = new_beliefs[i];
    }
    initializeTheta();
}
