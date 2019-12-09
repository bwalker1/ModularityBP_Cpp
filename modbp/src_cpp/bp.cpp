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

struct edge_data
{
public:
    index_t target;
    index_t layer;
    bool type;
    double weight;

    edge_data(index_t _target, index_t _layer, bool _type, double _weight) : target(_target), layer(_layer),  type(_type), weight(_weight) {};
};

BP_Modularity::BP_Modularity(const vector<vector<index_t>>& _layer_membership, const vector<pair<index_t,index_t> > &intra_edgelist, const vector<pair<double,double>> &intra_edgeweight,const vector<double> &inter_edgeweight, const vector<pair<index_t,index_t> > &inter_edgelist, const index_t _n, const index_t _nlayers,  const int _q, const index_t _num_biparte_classes, const double _beta, const vector<index_t>& _bipartite_class,  const double _omega, const double _dumping_rate, const double _resgamma, bool _verbose, bool _transform) :  layer_membership(_layer_membership), bipartite_class(_bipartite_class),neighbor_count(_n), neighbor_count_interlayer(_n), node_strengths(_n), theta(_nlayers),theta_bipartite(_num_biparte_classes), num_edges(_nlayers), n(_n), nlayers(_nlayers), num_biparte_classes(_num_biparte_classes), q(_q), beta(_beta), omega(_omega), dumping_rate(_dumping_rate), resgamma(_resgamma), verbose(_verbose), transform(_transform), order(_n), rng(time(NULL))
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
    bool is_bipartite = false;
    //fprintf(stdout,"is bipartite is not\n");

    if (num_biparte_classes>1 and ! bipartite_class.empty())
    {
        is_bipartite = true;
        fprintf(stdout,"setting is_bipartite\n");
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
    // TODO: go through all input edges and put them into the data structure along with categorization of their edge type
    

    for (index_t k=0;k<intra_edgelist.size();++k)
    {
        auto p = intra_edgelist[k];
        double w;
        index_t l;
//        if (weighted)
//        {
            //l gives the layer that the edge represents
        l = (index_t) intra_edgeweight[k].first;//typecast to index
        w = intra_edgeweight[k].second;

//        else
//        {
//            w = 1;
//        }
        index_t i = p.first;
        index_t j = p.second;
        edges[i].push_back(edge_data(j,true,l,w));
        edges[j].push_back(edge_data(i,true,l,w));
        
        num_edges[l] += w;
//        num_strength[layer_membership[i]] += w;

        total_strength += w;
        total_edges += 2;
        if (i!=j) { total_belief_edges+=2;}
    }
    
    for (index_t k=0;k<inter_edgelist.size();++k)
    {

        auto p = inter_edgelist[k];
        double w;

        w = inter_edgeweight[k];

        index_t i = p.first;
        index_t j = p.second;
        //We don't store layer information on the interlayer edges
        //We do store weight however
        edges[i].push_back(edge_data(j,false,-1,w));
        edges[j].push_back(edge_data(i,false,-1,w));
        neighbor_count_interlayer[i]+=1;
        neighbor_count_interlayer[j]+=1;
        total_edges += 2;
        if (i!=j) { total_belief_edges+=2;}
    }


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

        beliefs_offsets[i+1] = beliefs_offset_count;
        neighbors_offsets[i+1] = neighbors_offset_count;

        max_degree = max(max_degree,(index_t) edges[i].size());
        c_strength=0;
        node_strengths[i].resize(nlayers);

        for (index_t j=0;j<edges[i].size();++j)
        {

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
                neighbors[neighbor_c++] = edges[i][j].target;
                neighbor_offset_map[i][edges[i][j].target] = j;
            }

        }
        //all intralayer edges contribute to strength

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
    //fprintf(stderr,"Finished primary initialization\n");
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
    vector<double> c_strength = node_strengths[i];

    vector<index_t> cur_layers = layer_membership[i];
    // iterate over all states
    double Z = 0;
    for (index_t s = 0; s < q; ++s)
    {
        marginals[q*i+s] = 0;
        for (index_t idx2 = 0; idx2<nn; ++idx2)
        {
            bool type = neighbors_type[neighbors_offsets[i]+idx2];
            double add;
//            if (type==true)
//            {
                // intralayer contribution
//                if (weighted)

            add = log(1+scaleEdges[neighbors_offsets[i]+idx2]*(beliefs[beliefs_offsets[i]+nn*s+idx2]));

//                else
//                {
//                    add = log(1+scale*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
//                }
//            }
//            else
//            {
//                // interlayer contribution
//                add = log(1+scaleOmega*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
//            }
            marginals[q*i+s] += add;
        }
        // evaluate the rest of the update equation
        double field;
        if (is_bipartite) // bipartite case for single layer (each class has it's own theta)
            {
            index_t bpclass=bipartite_class[i];
            for (index_t lay=0;lay<nlayers;lay++)
                {
                field = c_strength[lay]*theta_bipartite[bpclass][s];
                }
            }
         else
            {
            for (index_t lay=0;lay<nlayers;lay++)
                {
                field=c_strength[lay]*theta[lay][s];
                }
            }

        double temp_inside = field + marginals[q*i+s];
        marginals[q*i+s] = exp(temp_inside);
        
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
        //We update nodes in a random order every other step.


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
        if (fast_convergence)
        {
            if (local_change < eps)
            {
                // not enough change in incoming beliefs to warrant an update
                continue;
            }
        }
        // if we changed any nodes, set this to true so we know we haven't converged
        //changed = true;
        change += local_change;
        
        // we should update the nodes contribution to theta
        compute_marginal(i);
        for (index_t s = 0; s < q; ++s)
        {
            if (is_bipartite)
            {
                index_t bpclass = bipartite_class[i];
                for(index_t c = 0; c<num_biparte_classes; ++c){
                    for (index_t lay=0;lay<nlayers;lay++)
                    {
                    if (c!=bpclass) //each node only contributes to null models outside of it's class
                        {
                        theta_bipartite[c][s] += -beta*resgamma/(total_strength)* c_strength[lay] * (marginals[q*i + s] - marginals_old[q*i + s]);
                        }
                    }

                }
            }
            else
            {
               for(index_t lay=0;lay<nlayers;lay++)
                    {
                 //each node contributes to each layer's theta according to edges it has in that layer
                theta[lay][s] += -beta*resgamma/(2*num_edges[lay])* c_strength[lay] * (marginals[q*i + s] - marginals_old[q*i + s]);
                    }
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
            // figure out the sum of logs part of the update equation that uses the incoming beliefs
            for (index_t idx=0; idx<nn; ++idx)
            {
                scratch[nn*s+idx] = 0;
                for (index_t idx2 = 0;idx2<nn;++idx2)
                {

                    bool type = neighbors_type[neighbors_offsets[i]+idx2];
                    double add;
//                    if (type==true)

                        // intralayer contribution

                    if (neighbors_offsets[i]+idx2 >= scaleEdges.size())
                        {
                          fprintf(stderr,"index violation: %d %d\n",neighbors_offsets[i]+idx2, scaleEdges.size());
                        }
                    //omega already folded into scaleEdges
                    add = log(1+scaleEdges[neighbors_offsets[i]+idx2]*(beliefs[beliefs_offsets[i]+nn*s+idx2]));

//                    else
//                    {
//                        // interlayer contribution
//                        add = log(1+scaleOmega*(beliefs[beliefs_offsets[i]+nn*s+idx2]));
//                    }
//                    scratch[nn*s+idx] += add;
                }
                // evaluate the rest of the update equation
//                printf("cscratch: %.3f , c_strength: %.3f, theta[t][s]: %.3f\n",scratch[nn*s+idx],c_strength,theta[t][s]);

                double field=0;

                if (is_bipartite) // bipartite case for single layer (each class has it's own theta)
                    {index_t bpclass=bipartite_class[i];
                    for(index_t lay=0;lay<nlayers;lay++)
                        {
                        field = c_strength[lay]*theta_bipartite[bpclass][s];
                        }
                    }
                 else
                    {
                    for(index_t lay=0;lay<nlayers;lay++)
                        {
                        field+=c_strength[lay]*theta[lay][s];
                        }
                    }

                double temp = exp(field + scratch[nn*s+idx]);
                scratch[nn*s+idx] = temp;
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
                bool type = neighbors_type[neighbors_offsets[i]+idx];
                double scaleHere;
//                if (type)

//                    if (weighted)

                scaleHere = scaleEdges[neighbors_offsets[i]+idx];

//                    else
//                    {
//                        scaleHere = scale;
//                    }

//                else
//                {
//                    scaleHere = scaleOmega;
//                }

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
            assert(sum==1);
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
        beta=eps; //must be non-zero.  make very small
    }
    scale = exp(beta)-1;
    
//    if (weighted)
//    {
    for (index_t i=0;i<total_edges;++i)
    {
            //omega has already been baked into edge weights for interlayer
            scaleEdges[i] = exp(beta*edge_weights[i])-1;
    }
//    }
    
    //scaleOmega = exp(beta*omega)-1;


    if (init_beliefs)
        initializeBeliefs();
    if (init_theta)
    {
        //fprintf(stdout,"is_bipartite2:%s\n", is_bipartite ? "true" : "false");
        if (is_bipartite)
                {
                //fprintf(stdout,"call to initialize bipartite theta\n");
                initializeTheta_bipartite();}
        else
                {initializeTheta();}
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

void BP_Modularity::initializeTheta_bipartite() {

    fprintf(stdout,"initializing bipartite theta.\n");

    //this is in the case of the bipartite graph where we have a
    //different theta for each class of node

    //TODO: This only works with a bipartite (or multipartite) graph that is single layer

//    for (index_t t = 0; t <nlayers; ++t)  // for right now we only allow single layer bipartite
//    {
    theta_bipartite.resize(num_biparte_classes);
    for (index_t bpclass = 0; bpclass < num_biparte_classes ; ++ bpclass)
    {
    // make sure the size is correct
        theta_bipartite[bpclass].resize(q);
        for (index_t s = 0; s<q;++s)
        {
            theta_bipartite[bpclass][s] = beta*resgamma/(q);
        }
    }
    //}
    //compute marginals and zero these back out for now
    // why do these have to be zeroed?
    compute_marginals();
    for (index_t t = 0; t <nlayers; ++t)
    {
        for (index_t s=0;s<q;++s)
        {
            theta_bipartite[t][s]=0;

        }
    }

    for (index_t i=0;i<n;++i)
        {
            index_t cur = bipartite_class[i];
            vector<double> c_strength = node_strengths[i];
            for (index_t c=0; c<num_biparte_classes; ++c){

                if (c!=cur){
                    //node i only contributes to the theta of classes other than it's own
                    for (index_t s = 0; s<q; ++s)
                    {
                        for(index_t lay=0;lay<nlayers;lay++)
                            {
                            theta_bipartite[c][s] += c_strength[lay] * marginals[q*i + s];
                            }
                    }
                }
            }

        }


        for (index_t c = 0; c < num_biparte_classes; ++c)
            {
                // fold in prefactor to theta
                for (index_t s = 0; s<q;++s)
                {
                    theta[c][s] *= -(beta*resgamma/(total_strength));
                }
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
        vector<index_t> t = layer_membership[i];
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
    vector<size_t> c_layer_ind;

    for (index_t i = 0; i < n; ++i) //iterate through all nodes (n)
    {
        c_layer_ind=layer_membership[i];
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

//double s(double beta, double omega, double q, double c)
//{
//    double eb = exp(beta);
//    double ewb = exp(omega*beta);
//
//    double temp1 = ((eb-1)/(eb-1+q));
//    double temp2 = ((ewb-1)/(ewb-1+q));
//    return c*temp1*temp1 + 2*temp2*temp2;
//}
//
//double sp(double beta, double omega, double q, double c)
//{
//    double eb = exp(beta);
//    double e2b = exp(2*beta);
//
//    double ewb = exp(omega*beta);
//    double e2wb = exp(2*omega*beta);
//
//
//    double temp1 = eb - 1 + q;
//    double temp2 = ewb- 1 + q;
//
//    double dxlamr= 2*q*(e2b-eb)/(temp1*temp1*temp1);
//    double dxlamt= 2*q*(e2wb-ewb)/(temp2*temp2*temp2);
//
//
//    return c*dxlamr+2*dxlamt;
//}


//bstar and excess degree handled on python side
//double BP_Modularity::compute_excess_degree(bool use_strength ) //default is false
//{
//// compute the excess degree.  If use strength uses total strength for node
//    double d2 = 0;
//    double d = 0;
//    double ss=0;
//    for (int i=0;i<n;++i)
//    {
//        //double nn = neighbor_count[i];
//        for (index_t lay =0;lay<nlayers;lay++){
//
//            if (use_strength){
//                ss = node_strengths[i][lay];
//                }
//             else{
//                ss = neighbor_count[i];
//             }
//            d2 += ss*ss;
//            d += ss;
//        }
//    }
//    double c = d2/d - 1;
//    return c;
//
//}

//computed on python side
//double BP_Modularity::compute_bstar(double omega_in, int q_in){
//
//
//    double c = compute_excess_degree(false);
//    double average_weight=0;
//    double tot =0 ;
//    //calculate average weights including omega as weights for inter layer
//    for (index_t i=0;i<n;++i)
//    {
//        average_weight+=node_strengths[i];
//        average_weight+=omega_in*neighbor_count_interlayer[i];
//        tot+=neighbor_count[i];
//    }
//    average_weight/=tot;
////    printf("q_in = %d , c = %.3f , avg_weight= %.3f , omega_in = %.3f ,tot=%.3f \n",q_in,c,average_weight,omega_in,tot);
//    double bstar =  (1.0/average_weight)*log(q_in /(sqrt(c)-1) +1) ;
//    return bstar;
//
//}

//double BP_Modularity::compute_bstar(double omega_in,int q_in)
//{
//    // currently this assumes multiplex graph
//
//    // compute c - decide on the right way
//    // the simple average degree
//    //double c = accumulate(num_edges.begin(), num_edges.end(), 0.0)/n;
//
//    double c = compute_excess_degree();
//
//
//    // bisection/newton hybrid method
//    double xl=0, xr=1;
//    double xn;
//
//    // find bounding interval
//    while (s(xr,omega_in,q_in,c) < 1)
//    {
//        xr *= 2;
//    }
//
//    // start newton's from midpoint
//    xn = (xl+xr)/2;
//    double yn = s(xn,omega_in,q_in,c);
//    double ypn = sp(xn,omega_in,q_in,c);
//
//    int maxiters = 100;
//    for (int iters=0;iters<maxiters;)
//    {
//        // try a newton step
//
//        xn -= (yn - 1)/ypn;
//        yn = s(xn,omega_in,q_in,c);
//        ypn = sp(xn,omega_in,q_in,c);
//
//        // check if this is in our bounding interval
//        if (xl < xn && xn < xr)
//        {
//            // narrow our interval using newton point
//            if (yn > 1)
//            {
//                xr = xn;
//            }
//            else
//            {
//                xl = xn;
//            }
//        }
//        else
//        {
//            // narrow our interval using bisection
//            double xc = (xl + xr)/2;
//            if (s(xc,omega_in,q_in,c)>1)
//            {
//                xr = xc;
//            }
//            else
//            {
//                xl = xc;
//            }
//
//            // restart newton's at the new midpoint
//            xn = (xl + xr)/2;
//            yn = s(xn,omega_in,q_in,c);
//            ypn = sp(xn,omega_in,q_in,c);
//        }
//
//        // check for convergence
//        if (xr - xl < 1e-6)
//        {
//            return (xl + xr)/2;
//        }
//    }
//    return (xl+xr)/2;
//}
