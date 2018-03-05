//
//  bp.hpp
//  beliefprop
//
//  Created by Benjamin Walker on 2/19/18.
//  Copyright © 2018 Benjamin Walker. All rights reserved.
//

#ifndef bp_hpp
#define bp_hpp

#define TRANSFORM 1

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <random>

using namespace std;

typedef unsigned char byte;
typedef unsigned long index_t;

void print_array(index_t *arr, index_t n);

class BP_Modularity
{
public:
    // initialize with Erdos-Renyi random graph
    BP_Modularity(const index_t n, const double p, const int q, const double beta, bool simultaneous = true, bool transform = true);
    ~BP_Modularity();
    
    // run BP to convergence
    bool run();
    // run one pass of the belief propagation update
    void step();
    void stepNew();
    
    void print_beliefs();
    void print_beliefs(size_t limit);
    void print_marginals(size_t limit);
    // if true: simultaneous updates of all beliefs. if false: go through nodes one-by-one (random order)
    
    
    void compute_marginals();
    
    void save_rgb();
    
    index_t return5(vector<pair<index_t,index_t> > edgelist);
private:
    void normalize(double *beliefs, index_t i);
    
    vector<unordered_map<index_t,index_t> > neighbor_offset_map;
    vector<index_t> neighbor_count;
    
    // private variables
    double * beliefs;
    double * beliefs_new;     // for out-of-place updates
    size_t * beliefs_offsets;
    
    index_t * neighbors;
    index_t * neighbors_reversed;
    size_t * neighbors_offsets;
    
    double * scratch;
    
    double * marginals;
    
    index_t n;
    int q;
    double beta;
    
    bool simultaneous;
    bool transform;
    vector<index_t> isomorphism;
    vector<index_t> r_isomorphism;
    
    index_t max_degree;
    
    unsigned long num_edges;
    
    double change;
    vector<double> changes;
    vector<index_t> order;
    
    double scale;
    
    double eps;
    
    bool changed;
    bool computed_marginals;
    bool save;
    
    inline index_t n_neighbors(index_t i) { return neighbors_offsets[i+1]-neighbors_offsets[i]; }
    
    void print_all_arrays();
    void print_neighbors(index_t k) { print_array(neighbors+neighbors_offsets[k],n_neighbors(k));}
    
    default_random_engine rng;
    
    vector<vector<byte> > reds,greens,blues;
};

#endif /* bp_hpp */
