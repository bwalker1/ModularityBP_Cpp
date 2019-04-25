//
//  bp.hpp
//  beliefprop
//
//  Created by Benjamin Walker on 2/19/18.
//  Copyright © 2018 Benjamin Walker. All rights reserved.
//

#ifndef bp_hpp
#define bp_hpp

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <random>
#include <time.h>

using namespace std;

typedef unsigned char byte;
typedef unsigned long index_t;

void print_array(index_t *arr, index_t n);

class BP_Modularity
{
public:
    // initialize from two edgelists: one giving intra-layer connections and another giving inter-layer connections, and also a list of which layer each node is in
	BP_Modularity(const vector<index_t> &layer_membership, const vector<pair<index_t, index_t> > &intra_edgelist, const vector<double> &intra_edgeweight, const vector<pair<index_t, index_t> > &inter_edgelist, const index_t _n, const index_t _nt, const int q, const double beta, const double omega = 1.0, const double resgamma = 1.0, bool verbose = false, bool transform = false);

    // run BP to convergence
    long run(unsigned long maxIters=100);

    // run one pass of the belief propagation update
    void step();

    void compute_marginals();
    double compute_bethe_free_energy();
    double compute_factorized_free_energy();

    vector<vector<double> > return_marginals();

    // accessors
    double getBeta() const { return beta; };
    void setBeta(double in, bool reset=true);

    // accessors
    double getResgamma() const { return resgamma; };
    void setResgamma(double in, bool reset=true);

    //accessors for omega
    double getOmega() const { return omega; };
    void setOmega(double in, bool reset=true);

    index_t getq() const { return q; };
    void setq(double new_q);

    void set_compute_bfe(const bool b) { compute_bfe = b; }

    bool getVerbose() const { return verbose; };
    void setVerbose(bool in) { verbose = in; };

    double compute_excess_degree(bool use_strength = false);
    double compute_bstar(double omega_in, int q_in);

    void permute_beliefs(vector<vector<index_t> > permutation);
    void merge_communities(vector<index_t> merges);

private:
    bool weighted;

//    void shuffleBeliefs(vector<vector<double>> in_beliefs);
    void initializeBeliefs();
    void initializeTheta();
    void normalize(vector<double> & beliefs, index_t i);
    void reinit(bool init_beliefs=true, bool init_theta=true);

    void compute_marginals(bool do_bfe_contribution);

    vector<unordered_map<index_t,index_t> > neighbor_offset_map;
    vector<index_t> neighbor_count;
    vector<index_t> neighbor_count_interlayer;

    vector<double> node_strengths;
    vector<double> edge_weights;

    // private variables
    vector<double> beliefs;
    vector<double> beliefs_old;     // for out-of-place updates
    vector<size_t> beliefs_offsets;

    vector<index_t> neighbors;
    vector<index_t> neighbors_reversed;
    vector<size_t> neighbors_offsets;
    vector<index_t> neighbors_type;

    vector<bool> connection_type;

    vector<double> scratch;

    vector<double> marginals;
	vector<double> marginals_old;

    vector< vector<double> > theta;

    vector<index_t> layer_membership;

    index_t n, nt;
    int q;
    double beta;
    double resgamma;
    double omega;

    bool transform;

    index_t max_degree;

    double bfe;
    bool compute_bfe;

    // vector containing total number of edges in each layer
    vector<unsigned long> num_edges;
//    vector<double>num_strength; //vector of total strength(weighted degree) each layer
    // sum of num_edges

    unsigned long total_edges;
    double total_strength;

    double change;
    vector<double> changes;
    vector<index_t> order;

    double scale, scaleOmega;
    vector<double> scaleEdges;  // weighted graph has scale varying for each edge
    double prefactor;
    
    double eps;
    
    bool changed;
    bool computed_marginals;
    
    inline index_t n_neighbors(index_t i) { return (index_t) neighbors_offsets[i+1]-neighbors_offsets[i]; }
    
    default_random_engine rng;

	void compute_marginal(index_t i, bool do_bfe_contribution = false);
    
    bool verbose;
    
    unsigned long iter;
};

#endif /* bp_hpp */
