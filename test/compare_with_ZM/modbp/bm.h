/*
 *   mod version 1.1, release date 03/18/2014
 *   Copyright 2014 Pan Zhang ( pan@santafe.edu )
 *   mod is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or 
 *   (at your option) any later version.

 *   sbm is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
*/
//{{{ defines and header files
#ifndef __BM__
#define __BM__
#define Q_PERMU 8 
#define MYEPS 1.0e-50
#define myexp exp 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <ctime>
#include <map>
#include <algorithm>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sstream>
#include <getopt.h>
#include "zrg.h"
#define FRANDOM (rg->rdflt())
const char deli[1024]="\t ";
//}}}
using namespace std;
long get_cpu_time(void);
//{{{class tnode
class tnode
{
	// class of nodes in dendogram, note that this is a node in the tree, contains set of real nodes of the graph
	public:
		vector <int> nodes; //set of nodes (in graph) in the tnode
		int idx; //index of this node in this level
		int level; //level of tnode in dendogram
		int father_idx; //index of its father in last level of dendogram. If tnode is root, father = -1;
		vector <int> children_idx; //index of children in next level of dendogram. If tnode is a leave, child is an empty set.
		vector <int> nodes_assign;
		tnode * father; //pointer to father;
		vector <tnode *> children; //pointer to children
		vector < vector <int> > graph_neis, graph_edges, groups_infer;
		vector < int > conf_infer;

		void build_children( vector <vector <tnode *> > &dendo, queue < tnode * > &to_select);
};
//}}}

class modbp
{
public:
	ZRANDOMv3 *rg;
	//{{{ variables
	vector < vector<int> > perms;//permutations
	int vflag;
	int LARGE_DEGREE;
	bool full;//used in BP (will be in nmf, which is not implemented yet) to avoid using further simplification than Bethe approximation)
	bool lin;//use linearized bp?
	bool parallel;
	double free_energy;
	double entropy;
	int randseed,randseed2;
	double bp_conv_crit;
	double beta; //for finite temperature inference
	double beta_sg;//spin glass transition;
	double expbeta,expmb;
	bool averr;
	int cut;
	double modularity;
	double ncut;
	double ftcin;
	double nonedge_ratio;
	double initbeta;
	int Q_max;

	// retrieval, select, hiera 
	bool retrieval_state;
	double retrieval_modularity;
	vector <vector <tnode *> > dendo; //dendogram containing class tnode.
	bool opt;//optimize modularity

	//graph part
	int N, Q, Q_true, M,TM, graph_max_degree;
	// N: number of nodes.  Q: number of colors of the model!!!   Q_true: number of colors given
	// M: number of edges	graph_max_degree: maximum degree of the graph.
	vector <string> graph_ids;//it stores id (in string) of nodes.
	vector < vector <int> > graph_neis, graph_neis_inv, graph_edges, graph_edges_missing,graph_neis_missing;
	// graph_neis[i][j] is the number of the j^th neighbor of i
	// graph_neis_inv[i][j] is the neighboring number of spin graph_neis[i][j] that correspond to i
	// graph_edges[] stores all (undirected!) edges of the graph.
	vector < vector<int> > groups_true,groups_infer;
	vector < int > conf_true,conf_infer;// conf_true stores configuration(assignment, or colors or communities) that given, and conf_infer stores configuration that infered.
	vector <double> graph_di;
	vector <double> exptmp;
	vector< vector <int> > graph_Aij;//adjacent matrix
	double fix;//fraction of variables that we fix to true configuration
	vector <double> conf_fix;

	//block model
	vector < double > na, nna, na_true, na_expect, nna_expect, eta, logeta, argmax_marginals; //nna is normalized na for degree corrected model.

	//learning and inference
	vector <double> h;
	vector <double> h2;
	int bp_last_conv_time;
	double bp_last_diff;
	vector <double> field_iter, normtot_psi, pom_psi, exph, maxpom_psii_iter;
	vector < vector < vector <double> >  > psi;  // psi[i][j] is the message from j-> i
	vector < vector < vector <double> >  > psi_new;  // used in parallel updating;
	vector < vector <double> > psii, real_psi, psii_iter; // real_psi is the total marginal of spin i

	//random number generator
	int seed;
	unsigned myrand, ira[256];
	unsigned char ip, ip1, ip2, ip3;
	//}}}
	//{{{ functions
	//permutation
	void init_perms();

	//graph
	void graph_build_neis_inv();
	void graph_read_graph(string);
	void graph_read_graph_spm(string);
	void graph_write_gml_infer(const char *);
	void graph_add_edge(int i, int j);

	//block model
	modbp(ZRANDOMv3 *rg_, int LARGE_DEGREE_, double bp_conv_crit_, int Q_, int vflag_, double beta_, double averr_,bool opt_);
	void set_Q(int);
	void set_vflag(int);
	void mod_record_conf_true(string);
	void mod_record_conf_infer(string);

	//message passing
	void bp_init_h();
	void bp_init(int);
	void bp_allocate();
	double mod_bp_compute_f();
	double mod_bp_compute_f_large_degree();
	double mod_bp_iter_update_psi(int, double);
	double mod_bp_iter_update_psi_large_degree(int, double);
	int bp_converge(double, int, int, double);
	void bp_set_true_message(int i);
	void bp_set_random_message(int i);
	bool check_factorized_solution(double);

	void compute_argmax_conf();
	double compute_overlap();
	double compute_overlap_marg();
	double compute_overlap_fraction();
	double compute_config_ovl();
	double compute_overlap_EA();
	void compute_cuts();
	void form_groups_infer();

	//Output part
	void show_marginals(int );
	void output_marginals(string );
	void output_group_sizes();
	void output_f_ovl(int);

	void infer2true();

	void do_inference(double, int, double);
	bool retrieval(double, int, double, int);//find retrieval state given Q
	bool sg(double, int, double, int);//find spin glass transition point
	int select(double, int, double,int);//select a Q using retrieval modularity
	void shuffle_seq(vector <int> &sequence);

	//hiera
	void hiera(double, int, double, int);
	void hiera2(double, int, double, int);
	void print_dendo(string);
	void copy_struct(tnode *);
};
//}}}
string get_std_from_cmd(const char*);
vector <string> strsplit(const string& , const string&);


#endif

