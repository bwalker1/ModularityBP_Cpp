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
#include "bm.h"
#include <sstream>
#include <cstring>
#include <string>
//{{{void modbp::shuffle_seq(vector <int> &sequence)
void modbp::shuffle_seq(vector <int> &sequence)
{
	//random shuffle the sequence
	//I put this function inside class modbp because I need to use the random number generator inside the class.
	for(int i=0;i<N;i++) sequence[i]=i;
	int tmp;
	for(int i=0;i<N;i++){
		int tmpindex=int(FRANDOM*N);
		//int tmpindex=int(rg->rdflt()*(N-i));
		tmp=sequence[tmpindex];
		sequence[tmpindex]=sequence[N-i-1];
		sequence[N-i-1]=tmp;
	}
}
//}}}
//set parameters
//{{{modbp::modbp(ZRANDOMv3 *rg_, int LARGE_DEGREE_, double bp_conv_crit_, int Q_, int vflag_, double beta_, double averr_,bool opt_)
modbp::modbp(ZRANDOMv3 *rg_, int LARGE_DEGREE_, double bp_conv_crit_, int Q_, int vflag_, double beta_, double averr_,bool opt_)
{ 
	rg = rg_;
	LARGE_DEGREE = LARGE_DEGREE_;
	bp_conv_crit = bp_conv_crit_;
	Q = Q_;
	init_perms();
	vflag = vflag_;
	beta = beta_;
	averr = averr_;
	opt = opt_;
}
//}}}
//graph part
//{{{ void modbp:: graph_read_graph(string fname)
void modbp:: graph_read_graph(string fname)
{
	//read from file "fname" to build graph_ids[] graph_neis[][] graph_edges[] graph_neis_inv[][] groups_true[][]
	//and set N M and graph_max_degree
	string snode=string("node"),sbegin=string("["), send=string("]"), sedge=string("edge"),sid=string("id"),svalue=string("value"),ssource=string("source"),starget=string("target");
	map <string,int> id2idx;
	map <string, string> id2value;
	string tmpstr;
	ifstream fin(fname.c_str());
	graph_ids.clear();
	groups_true.clear();
	map <string, int> value2color;
	cout<<"reading "<<fname<<"... "<<flush;
	assert(fin.good()&&"I can not open the file that contains the graph.");
	int idx=0,color_idx=0;
	while(fin >> tmpstr){ // read something if it is possible to read all nodes and build node structures.
		if(tmpstr == snode){//node section begins, let's read id and value.
			fin >> tmpstr;
			assert(tmpstr == sbegin && "[ should follow node");
			bool endnode=false;
			string myid("not_set");
			for(int readtimes=0;readtimes<100;readtimes++){//read at most 100 times to scan all parameters of this node
				fin >> tmpstr;
				if(tmpstr == send) {
					endnode=true;
					break;
				}else if(tmpstr == sid){//read an id and assign an index to this id
					fin >> myid;
					assert(id2idx.count(myid) == 0&&"multi-definition of node in graph file!");
					id2idx[myid] = idx++;
					graph_ids.push_back(myid);
				}else if(tmpstr == svalue){//read a value to an id
					assert(myid != string("not_set") && "id should be given before value");
					fin >> tmpstr;
					id2value[myid] = tmpstr;
					if(value2color.count(tmpstr) == 0){
						vector <int> tmpvec; tmpvec.clear();
						groups_true.push_back(tmpvec);
						value2color[tmpstr] = color_idx++;
					}
				}else{
					//cout<<"this is "<<tmpstr<<" which should be ignored"<<endl;	
				}
			}
			assert(endnode && "there must be somthing wrong because I can not find ]");
		}
	}	
	fin.close();
	N=graph_ids.size();

	graph_neis.resize(N);
	graph_di.resize(N);
	for(int i=0;i<N;i++) graph_neis[i].resize(0);
	graph_edges.resize(0);
	fin.open(fname.c_str());
	int i=-1,j=-1;
	while(fin >> tmpstr){ // read again the file for edges and to build edge structrues.
		if(tmpstr == sedge){//edge section begins, let's read pairs of nodes
			fin >> tmpstr;
			assert(tmpstr == sbegin && "[ should follow edge");
			fin >> tmpstr;
			while(tmpstr != ssource) fin>>tmpstr;
//			assert(tmpstr == ssource && "there should be souce following edge [");
			fin >> tmpstr;
			i=id2idx[tmpstr];
			fin >> tmpstr;
			assert(tmpstr == starget && "there should be target following source");
			fin >> tmpstr;
			j=id2idx[tmpstr];
			assert(i<N && i>=0 && j<N && j>=0 && "head of this edge does not exist in the node database!");
			bool duplicate=false;
			for(int idxij=0;idxij<graph_neis[i].size();idxij++){
				if(graph_neis[i][idxij] == j){
					duplicate = true;
					break;
				}
			}
			if( !duplicate ){
				for(int idxji=0;idxji<graph_neis[j].size();idxji++){
					if(graph_neis[j][idxji] == i){
						duplicate = true;
						break;
					}
				}
			}
			//assert(!duplicate && "there are duplicated edges! Make sure the file is OK");
			if( !duplicate){
				graph_add_edge(i,j);
			}
		}
	}
	fin.close();
	M=graph_edges.size();
	TM=M*2;
	Q_true=groups_true.size();

	cout<<graph_ids.size()<<" nodes "<<groups_true.size()<<" groups, "<<graph_edges.size()<<" edges, c="<<2.0*M/N<<flush;

	graph_build_neis_inv();//build neighbor matrix graph_neis[][] and graph_neis_inv[][] from edge matrix graph_edges.
	conf_true.resize(N);//true assignments
	for(map<string,string>::iterator it=id2value.begin();it != id2value.end(); ++it){
		int idx=id2idx[it->first];
		int color=value2color[it->second];
		conf_true[idx] = color;
		groups_true[color].push_back(idx);
	}
	graph_max_degree=-50;
	for(int i=0; i<N; i ++){
		if(int(graph_neis[i].size()) >= graph_max_degree) graph_max_degree=graph_neis[i].size();
	}
	for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size();
	cout<<" max_degree="<<graph_max_degree<<endl;
}
//}}}
//{{{ void modbp:: graph_read_graph_spm(string fname)
void modbp:: graph_read_graph_spm(string fname)
{
	//read from file "fname" to build graph_ids[] graph_neis[][] graph_edges[] graph_neis_inv[][] groups_true[][]
	//and set N M and graph_max_degree
	//this function will read fname three times. The first time to determine type of the graph, second 
	//time for number of nodes and third time for all the edges
	string tmpstr;
	ifstream fin(fname.c_str());
	graph_ids.clear();
	groups_true.clear();
	cout<<"reading "<<fname<<"... "<<flush;
	assert(fin.good()&&"I can not open the file that contains the graph.");
	
	char line[1024];
	fin.getline(line,1024,'\n');
	int nx=0;
	char *res=NULL;
	res=strtok(line, " ");
	while (res != NULL){
		res=strtok(NULL, " ");
		nx++;
	}
	cout<<"number of variables in one line is "<<nx<<endl;
	assert(nx==2 && "One line should contain 2 variables");
	fin.close();

	fin.open(fname.c_str());
	bool min_one=false;
	int idx=0;
	int minidx=-1, maxidx=-1;
	while(fin >> idx){ // read first to determine number of nodes
		if(minidx == -1 || maxidx == -1) {
			if(minidx == -1) minidx=idx;
			if(maxidx == -1) maxidx=idx;
		}else{
			if(idx > maxidx) maxidx=idx;
			if(idx < minidx) minidx=idx;
		}
	}
	assert(minidx >= 0 && maxidx >= 0);
	if( minidx == 1) min_one=true; //index starting from 1
	fin.close();
	if(min_one) N=maxidx;
	else N=maxidx+1;
	fin.close();

	graph_neis.resize(N);
	graph_di.resize(N);
	for(int i=0;i<N;i++) graph_neis[i].resize(0);
	graph_edges.resize(0);
	fin.open(fname.c_str());
	int i=-1,j=-1;
	while(!fin.eof()){
		if(fin>>i && fin >>j){
			if(min_one){
				i -= 1;
				j -= 1;
			}
			assert(i<N && i>=0 && j<N && j>=0 && "head of this edge does not exist in the node database!");
			//cout<<i<<" "<<j<<endl;
			bool duplicate=false;
			for(int idxij=0;idxij<graph_neis[i].size();idxij++){
				if(graph_neis[i][idxij] == j){
					duplicate = true;
					break;
				}
			}
			if( !duplicate ){
				for(int idxji=0;idxji<graph_neis[j].size();idxji++){
					if(graph_neis[j][idxji] == i){
						duplicate = true;
						break;
					}
				}
			}
			if( !duplicate){
				graph_add_edge(i,j);
			}
		}
	}
	fin.close();
	M=graph_edges.size();
	TM=M*2;
	cout<<graph_neis.size()<<" nodes "<<graph_edges.size()<<" edges, c="<<2.0*M/N<<flush;

	graph_build_neis_inv();//build neighbor matrix graph_neis[][] and graph_neis_inv[][] from edge matrix graph_edges.
	graph_max_degree=-50;
	for(int i=0; i<N; i ++){
		if(int(graph_neis[i].size()) >= graph_max_degree) graph_max_degree=graph_neis[i].size();
	}
	for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size();
	cout<<" max_degree="<<graph_max_degree<<endl;
}
//}}}
//{{{void modbp:: graph_add_edge(int i, int j)
void modbp:: graph_add_edge(int i, int j)
{
	graph_neis[i].push_back(j);
	for(int idxj=0;idxj<graph_neis[i].size()-1;idxj++){
		if(graph_neis[i][idxj]>graph_neis[i][graph_neis[i].size()-1]){
			int xtmp=graph_neis[i][idxj];
			graph_neis[i][idxj]=graph_neis[i][graph_neis[i].size()-1];
			graph_neis[i][graph_neis[i].size()-1]=xtmp;
		}
	}
	graph_neis[j].push_back(i);
	for(int idxi=0;idxi<graph_neis[j].size()-1;idxi++){
		if(graph_neis[j][idxi]>graph_neis[j][graph_neis[j].size()-1]){
			int xtmp=graph_neis[j][idxi];
			graph_neis[j][idxi]=graph_neis[j][graph_neis[j].size()-1];
			graph_neis[j][graph_neis[j].size()-1]=xtmp;
		}
	}
	vector<int> tmpvec; tmpvec.clear();
	tmpvec.push_back(i);
	tmpvec.push_back(j);
	graph_edges.push_back(tmpvec);
}
//}}}
//{{{ void modbp:: graph_build_neis_inv()
void modbp:: graph_build_neis_inv()
{
	//build graph_neis_inv
	graph_neis_inv.resize(N);
	for (int i=0;i<N;i++){
		graph_neis_inv[i].resize(graph_neis[i].size());
		for (int idxij=0;idxij<graph_neis[i].size();idxij++){
			int j=graph_neis[i][idxij];
			int target=0;
			for (int idxji=0;idxji<graph_neis[j].size();idxji++){
				if (graph_neis[j][idxji]==i){
					target = idxji;
					break;
				}
			}
			graph_neis_inv[i][idxij]=target;
		}
	}
}
//}}}
//{{{void modbp:: graph_write_gml_infer(char *fname)
void modbp:: graph_write_gml_infer(const char *fname)
{
	ofstream fout(fname);
	assert(fout.good() && "can not write to the file");
	cout<<"Writting graph into "<<fname<<"..."<<flush;
	fout << "graph [" << endl;
	for(int i=0;i<N;i++){
		fout << "  node" << endl;
		fout << "  [" << endl;
		fout << "    id " << i << endl;
		if(conf_infer.size() == N) fout << "    value " << conf_infer[i] << endl;
		fout << "  ]"<< endl;
	}
	for(int i=0;i<M;i++){
		fout << "  edge" << endl;
		fout << "  [" << endl;
		fout << "    source " << graph_edges[i][0] << endl;
		fout << "    target " << graph_edges[i][1] << endl;		
		fout << "  ]" << endl;
	}
	fout << "]" << endl;
	fout.close();
	cout<<"Done!"<<endl;
}
//}}}

//block model
//{{{void modbp:: infer2true()
void modbp:: infer2true()
{
	for(int i=0;i<N;i++) conf_true[i]=conf_infer[i];
}
//}}}
//{{{void modbp:: mod_record_conf_true(string fname)
void modbp:: mod_record_conf_true(string fname)
{
	ofstream fout(fname.c_str());
	assert(fout.good() && "can not write to the file");
	cout<<"Writting conf_true into "<<fname<<"..."<<flush;
	for(int i=0;i<N;i++) fout<<conf_true[i]<<endl;
	fout.close();
//	cout<<"Done!"<<endl;
}
//}}}
//{{{void modbp:: mod_record_conf_infer(string fname)
void modbp:: mod_record_conf_infer(string fname)
{
	ofstream fout(fname.c_str());
	assert(fout.good() && "can not write to the file");
//	cout<<"Writting conf_infer into "<<fname<<"..."<<flush;
	for(int i=0;i<N;i++) fout<<conf_infer[i]<<endl;
	fout.close();
//	cout<<"Done!"<<endl;
}
//}}}

//compute part
//{{{double modbp:: compute_overlap()
double modbp:: compute_overlap()
{
	double max_ov=-1.0,connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
	for (int pe=0;pe<perms.size();pe++){
		double ov=0.0;
		for (int i=0;i<N;i++) ov+=(real_psi[i][perms[pe][conf_true[i]]]);
		ov/=N;
#ifdef OVL_NORM
		ov=(ov-connect)/(1-connect);
#endif
		if (ov>max_ov) max_ov=ov;
	}
	return max_ov;
}
//}}}
//{{{double modbp:: compute_overlap_EA()
double modbp:: compute_overlap_EA()
{
	double ov=0.0;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++){
			ov += real_psi[i][q]*real_psi[i][q];
		}
	}
	ov /= N;
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
#ifdef OVL_NORM
	ov=(ov-connect)/(1-connect);
#endif
	return ov;
}
//}}}
//{{{double modbp:: compute_overlap_fraction()
double modbp:: compute_overlap_fraction()
{
	//cout<<check_factorized_solution(bp_conv_crit*10)<<endl;
	if(check_factorized_solution(bp_conv_crit*10)){
//		cout<<"factorized solution"<<endl;
		return 1.0/Q;
	}
	compute_argmax_conf();	
	double ovl=compute_config_ovl();
	return ovl;
}
//}}}
//{{{double modbp:: compute_config_ovl()
double modbp:: compute_config_ovl()
{
	double max_ov=-1.0;	
	init_perms();
	for(int pe=0;pe<perms.size();pe++){
		double ov=0.0;
		for(int j=0;j<N;j++){
			if(conf_true[j]==perms[pe][conf_infer[j]]) ov += 1.0;
		}
		ov /= N;
		if (ov>max_ov) max_ov=ov;
	}
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
#ifdef OVL_NORM
	max_ov=(max_ov-connect)/(1-connect);
#endif
	compute_cuts();
	return max_ov;
}
//}}}
//{{{double void:: compute_argmax_conf()
void modbp:: compute_argmax_conf()
{
	for(int i=0;i<N;i++){
		argmax_marginals[i]=real_psi[i][0];
		conf_infer[i]=0;
		for(int q=0; q<Q; q++){
			if(real_psi[i][q]>argmax_marginals[i]){
				argmax_marginals[i]=real_psi[i][q];
				conf_infer[i]=q;
			}
		}
	}
	form_groups_infer();
}
//}}}
//{{{void modbp:: form_groups_infer()
void modbp:: form_groups_infer()
{
	groups_infer.resize(Q);
	for(int q=0;q<Q;q++) groups_infer[q].clear();
	for(int i=0;i<N;i++) groups_infer[conf_infer[i]].push_back(i);	
//	cout<<"#Group sizes:\t";
//	for(int q=0;q<Q;q++) cout<<groups_infer[q].size()<<" ";cout<<endl;
}
//}}}
//{{{void modbp:: output_group_sizes()
void modbp:: output_group_sizes()
{
	form_groups_infer();
	cout<<"#Group sizes:\t";
	for(int q=0;q<Q;q++) cout<<groups_infer[q].size()<<" ";cout<<endl;
//	cout<<"#Group average degree:"<<endl;
//	for(int q=0;q<Q;q++){
//		double ma=0;
//		for(int i=0;i<groups_infer[q].size();i++) ma += graph_neis[groups_infer[q][i]].size();
//		ma /= groups_infer[q].size();
//		cout<<ma<<" ";
//	}
//	cout<<endl;
}
//}}}
		
//{{{double modbp:: compute_overlap_marg()
double modbp:: compute_overlap_marg()
{
	compute_argmax_conf();
	double ov=0.0;
	for(int i=0;i<N;i++) ov += argmax_marginals[i];
	ov /= N;
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na[q])/N,2);
#ifdef OVL_NORM
	ov=(ov-connect)/(1-connect);
#endif
	return ov;
}
//}}}
//{{{void modbp:: compute_cuts()
void modbp:: compute_cuts()
{
	cut=0; ncut=0;
	for(int a=0;a<M;a++){
		if(conf_infer[graph_edges[a][0]] != conf_infer[graph_edges[a][1]]) cut ++;
	}
	double mod2=M-cut;
	if(check_factorized_solution(1.0e-4)){
		mod2=0;
	}else{
		for(int q=0;q<groups_infer.size();q++){
			double tmp=0;
			for(int idx=0;idx<groups_infer[q].size();idx++){
				int i=groups_infer[q][idx];
				tmp += graph_di[i];
			}
			mod2 -= (tmp/2.0/M)*(tmp/2.0);
		}
		mod2 /=M;
	}
	//cout<<"mod2="<<mod2<<endl;
//	modularity=M-cut;
//	for(int i=0;i<N;i++){
//		int di=graph_di[i];
//		for(int j=i+1;j<N;j++){
//			if(conf_infer[i] == conf_infer[j]) modularity -= di*graph_di[j]/2/M;
//		}
//	}
//	modularity /= M;
	//cout<<"modularity="<<modularity<<endl;
	modularity=mod2;
}
//}}}

//BP part
// {{{void modbp:: bp_allocate()
void modbp:: bp_allocate()
{
	conf_infer.resize(N);
	argmax_marginals.resize(N);
	psi.resize(N) ;
	exptmp.resize(Q);
	for(int i=0;i<N;i++){
		psi[i].resize(graph_neis[i].size());
		for (int idxij=0;idxij<graph_neis[i].size();idxij++){
			psi[i][idxij].resize(Q);
		}
	}
	pom_psi.resize(Q);
	psii.resize(Q);

	real_psi.resize(N) ;
	h.resize(Q);
	h2.resize(Q);
	exph.resize(Q,0);
	for (int i=0;i<N;i++) real_psi[i].resize(Q);
	psii_iter.resize(Q);
	maxpom_psii_iter.resize(graph_max_degree);
	for (int q=0;q<Q;q++) psii_iter[q].resize(graph_max_degree);
	field_iter.resize(graph_max_degree);
	normtot_psi.resize(graph_max_degree);
}
//}}}
//{{{double modbp:: mod_bp_compute_f()
double modbp:: mod_bp_compute_f()
{
	double totalf=0.0;
	double f_site=0;
	double c=2.0*M/N;
	for (int i=0;i<N;i++){
		double di=graph_di[i];
		double maxpom_psi=-1000000.0;
		for(int q=0;q<Q;q++){
			double a=0.0;
			double tmp=0.0;
			for (int l=0;l<graph_neis[i].size();l++){
				double b = 1.0-psi[i][l][q]+expbeta*psi[i][l][q];
				a += log(b);
				tmp += expbeta*psi[i][l][q]/b;
			}
			tmp -= di/2.0/M*h[q];
			pom_psi[q]=a-beta*di/TM*h[q];
			if(pom_psi[q] > maxpom_psi) maxpom_psi = pom_psi[q];
		}
		double normtot=0;
		for(int q=0;q<Q;q++) normtot += exp(pom_psi[q]-maxpom_psi);
		normtot = maxpom_psi+log(normtot);
		f_site += normtot;

		double theory=0;
		if(graph_di[i] == 0) theory = log(Q);
		else theory = log(Q) + di*log(1.0-1.0/Q+expbeta/Q)  - beta*graph_di[i]/Q;
	}
	f_site/=N;

	double theory_site=log(Q)+log(1-1.0/Q+expbeta/Q)*c-beta/Q*c;

	double f_link=0;
	for (int i=0;i<N;i++){
		for (int l=0;l<graph_neis[i].size();l++){
			double norm_L=0;
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double tmp=0;
			double p1=0,p2=0;
			for (int q1=0;q1<Q;q1++){
				tmp += expbeta*psi[i][l][q1]*psi[i2][l2][q1];
				for (int q2=0;q2<Q;q2++){
					if(q1==q2){
						p1 += psi[i][l][q1]*psi[i2][l2][q2];
						norm_L+=expbeta*psi[i][l][q1]*psi[i2][l2][q2];
					}else{
						norm_L+=psi[i][l][q1]*psi[i2][l2][q2];
						p2 += psi[i][l][q1]*psi[i2][l2][q2];
					}
				}
			}
			f_link+=log(norm_L);
		}
	}
	f_link/=2*N;
	double theory_link = log((1.0-1.0/Q+expbeta/Q))*c/2.0;

	double last_term=0;
	for (int q1=0;q1<Q;q1++) last_term += h[q1]*h[q1];
	last_term = -1.0*last_term*beta/4/M/N;
	double theory_last_term = -1.0*beta*M/Q/N;
	
	totalf=f_site-f_link-last_term;
	double lambda=(expbeta-1.0)/(1.0+expbeta/Q-1.0/Q)/Q;

	free_energy= -1.0/beta*totalf;
	return -1.0/beta*totalf;
}
//}}}
//{{{double modbp:: mod_bp_compute_f_large_degree()
double modbp:: mod_bp_compute_f_large_degree()
{
	double totalf=0.0;
	double f_site=0;
	double c=2.0*M/N;
	for (int i=0;i<N;i++){
		double di=graph_di[i];
		double maxpom_psi=-1000000.0;
		for(int q=0;q<Q;q++){
			double a=0.0;
			double tmp=0.0;
			for (int l=0;l<graph_neis[i].size();l++){
				double b = expmb-expmb*psi[i][l][q]+psi[i][l][q];
				a += log(b)+beta;
				tmp += expbeta*psi[i][l][q]/b;
			}
			tmp -= di/2.0/M*h[q];
			pom_psi[q]=a-beta*di/TM*h[q];
			if(pom_psi[q] > maxpom_psi) maxpom_psi = pom_psi[q];
		}
		double normtot=0;
		for(int q=0;q<Q;q++) normtot += exp(pom_psi[q]-maxpom_psi);
		normtot = maxpom_psi+log(normtot);
		f_site += normtot;


		double theory=0;
		if(graph_di[i] == 0) theory = log(Q);
		else theory = log(Q) + di*log(1.0-1.0/Q+expbeta/Q)  - beta*graph_di[i]/Q;
	}
	f_site/=N;

	double theory_site=log(Q)+log(1-1.0/Q+expbeta/Q)*c-beta/Q*c;

	double f_link=0;
	for (int i=0;i<N;i++){
		for (int l=0;l<graph_neis[i].size();l++){
			double norm_L=0;
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double tmp=0;
			double p1=0,p2=0;
			for (int q1=0;q1<Q;q1++){
				tmp += psi[i][l][q1]*psi[i2][l2][q1];
				for (int q2=0;q2<Q;q2++){
					if(q1==q2){
						p1 += psi[i][l][q1]*psi[i2][l2][q2];
						norm_L += psi[i][l][q1]*psi[i2][l2][q2];
					}else{
						norm_L+=expmb*psi[i][l][q1]*psi[i2][l2][q2];
						p2 += psi[i][l][q1]*psi[i2][l2][q2];
					}
				}
			}
			f_link+=log(norm_L)+beta;
		}
	}
	f_link/=2*N;
	double theory_link = log((1.0-1.0/Q+expbeta/Q))*c/2.0;

	double last_term=0;
	for (int q1=0;q1<Q;q1++) last_term += h[q1]*h[q1];
	last_term = -1.0*last_term*beta/4/M/N;
	double theory_last_term = -1.0*beta*M/Q/N;
	
	totalf=f_site-f_link-last_term;
	double lambda=(expbeta-1.0)/(1.0+expbeta/Q-1.0/Q)/Q;

	free_energy= -1.0/beta*totalf;
	return -1.0/beta*totalf;
}
//}}}
//{{{void modbp:: bp_init_h()
void modbp:: bp_init_h()
{
	h.resize(Q,0);
	for(int q=0;q<Q;q++) h[q]=0;
	for (int i=0;i<N;i++){
		double di=graph_di[i];
		// update h
		for (int q1=0;q1<Q;q1++){
			if(beta>999){
				double maxterm=-10000;
				for(int q2=0;q2<Q;q2++){
					double tmp=+real_psi[i][q2];
					if(tmp>maxterm) maxterm=tmp;
				}
			}else h[q1] += graph_di[i]*real_psi[i][q1];
		}
	}
	for(int q=0;q<Q;q++){
		if(beta<0) exph[q]=myexp(-h[q]/N);
		else exph[q]=myexp(h[q]);
	}
}
//}}}
//{{{void modbp:: bp_init(int initflag)
void modbp:: bp_init(int init_flag)
{
	for (int i=0;i<N;i++) bp_set_random_message(i);
	bp_init_h();
}
//}}}
//{{{double modbp::mod_bp_iter_update_psi(int i,double dumping_rate)
double modbp::mod_bp_iter_update_psi(int i,double dumping_rate)
{
	int l,q;
	double di=graph_di[i];
	TM=2.0*M;
	for (int q1=0;q1<Q;q1++) h[q1] -= di*real_psi[i][q1];
	double a=1.0, b=0.0, normtotal=0.0, normtot_real=0.0;
	for(int j=0;j<graph_neis[i].size();j++) normtot_psi[j]=0;
	for (q=0;q<Q;q++){
		a=1.0;
		for (l=0;l<graph_neis[i].size();l++){
			b = 1.0-psi[i][l][q]+expbeta*psi[i][l][q];
			a *= b;
			field_iter[l]=b;
		}
		pom_psi[q]=a*exp(-1.0*beta*di/TM*h[q]);
		normtot_real += pom_psi[q];
		for (l=0;l<graph_neis[i].size();l++){
			if(field_iter[l]<MYEPS){//to cure the problem that field_iter[l] may be very small
				double tmprob=exp(-1.0*beta*di/TM*h[q]);
				for(int lx=0;lx<graph_neis[i].size();lx++){
					if(lx==l) continue;
					tmprob *= field_iter[lx];
				}
				psii_iter[q][l]=tmprob;
			}else {
				psii_iter[q][l]=pom_psi[q]/(field_iter[l]);
			}
			normtot_psi[l]+= psii_iter[q][l];
		}
	}
	//for (int q1=0;q1<Q;q1++) h[q1] -= di*real_psi[i][q1];
	// normalization
	double mymaxdiff=-100000.0;
	int ecount=0;
	if(averr) mymaxdiff=0;
	for (q=0;q<Q;q++){
		real_psi[i][q]=pom_psi[q]/normtot_real;
		for (l=0;l<graph_neis[i].size();l++){
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double tmp=1.0;
			tmp=psii_iter[q][l]/normtot_psi[l];
			double mydiff=fabs(psi[i2][l2][q]-tmp);
			//if(isnan(mydiff)){
			//	cout<<endl<<psi[i2][l2][q]<<" "<<tmp<<" norm="<<normtot_psi[l]<<endl;
			//	exit(0);
			//}
			if(averr){
				mymaxdiff += mydiff;
				ecount++;
			}else{
				if(mydiff > mymaxdiff) mymaxdiff = mydiff;
			}
			psi[i2][l2][q]=(dumping_rate)*tmp+(1.0-dumping_rate)*psi[i2][l2][q];//update psi_{i\to j}
		}
	}
//	if(mymaxdiff<-1){
//		cout<<mymaxdiff<<" "<<graph_neis[i].size()<<endl;
//	}
	for (int q1=0;q1<Q;q1++) h[q1] += di*real_psi[i][q1];
	if(averr) return mymaxdiff/ecount;
	else return mymaxdiff;
}
//}}}
//{{{double modbp::mod_bp_iter_update_psi_large_degree(int i,double dumping_rate)
double modbp::mod_bp_iter_update_psi_large_degree(int i,double dumping_rate)
{
	int l,q;
	double di=graph_di[i];
	TM=2.0*M;
	for (int q1=0;q1<Q;q1++) h[q1] -= di*real_psi[i][q1];
	double a=0.0, b=0.0, normtotal=0.0, normtot_real=0.0,max_a=-1000.0;
	for(int j=0;j<graph_neis[i].size();j++) normtot_psi[j]=0;
	for (q=0;q<Q;q++){
		a=0.0;
		for (l=0;l<graph_neis[i].size();l++){
			//b = 1.0-psi[i][l][q]+expbeta*psi[i][l][q];
			//field_iter[l]=log(b);
			b = expmb-expmb*psi[i][l][q]+psi[i][l][q];
			field_iter[l]=log(b)+beta;
			a += field_iter[l];
		}
		pom_psi[q]=a-1.0*beta*di/TM*h[q];
		if(q==0) max_a=pom_psi[q];
		else{
			if(pom_psi[q]>max_a) max_a=pom_psi[q];
		}
		for(l=0;l<graph_neis[i].size();l++){
			psii_iter[q][l]=pom_psi[q]-field_iter[l];
		}
	}
	
	//computing marginals 
	normtot_real =0;
	for(q=0;q<Q;q++) {
		exptmp[q]=exp(pom_psi[q] - max_a);
		normtot_real += exptmp[q];
	}
	for(q=0;q<Q;q++) real_psi[i][q]=exptmp[q]/normtot_real;

	//computing cavity quantities:
	double mymaxdiff=-100000.0;
	int ecount=0;
	if(averr) mymaxdiff=0;
	for(l=0;l<graph_neis[i].size();l++){
		int i2=graph_neis[i][l];
		int l2=graph_neis_inv[i][l];
		for(q=0;q<Q;q++){
			if(q==0) max_a=psii_iter[q][l];
			else{
				if(psii_iter[q][l]>max_a) max_a=psii_iter[q][l];
			}
		}
		normtot_psi[l]=0;
		for(q=0;q<Q;q++) {
			exptmp[q]=exp(psii_iter[q][l] - max_a);
			normtot_psi[l] += exptmp[q];
		}
		for(q=0;q<Q;q++) {
			double tmp = exptmp[q]/normtot_psi[l];
			double mydiff=fabs(psi[i2][l2][q]-tmp);
			if(averr){
				mymaxdiff += mydiff;
				ecount++;
			}else{
				if(mydiff > mymaxdiff) mymaxdiff = mydiff;
			}
			psi[i2][l2][q]=(dumping_rate)*tmp+(1.0-dumping_rate)*psi[i2][l2][q];//update psi_{i\to j}
		}
	}

	for (int q1=0;q1<Q;q1++) h[q1] += di*real_psi[i][q1];
	if(averr) return mymaxdiff/ecount;
	else return mymaxdiff;
}
//}}}
//{{{int modbp::bp_converge(double bp_err, int max_iter_time, int init_flag, double dumping_rate)
int modbp::bp_converge(double bp_err, int max_iter_time,int init_flag, double dumping_rate)
{
	if(beta<-100){// use beta*
		double c=2.0*M/N;
		beta=log( Q/(sqrt(c)-1)+1);
		cout<<"beta is set to beta* = "<<beta<<endl;
	}
	bp_init(1);
	if(vflag >= 10) show_marginals(vflag);
	else if(vflag >= 3) show_marginals(10);
	bp_init_h();
	int iter_time=0;
	int bin=max_iter_time/20;
	for(iter_time=0; iter_time<max_iter_time;iter_time++){
		double maxdiffm = -100000.0;
		double old=1;
		double linc=M/N*2;
		for(int iter_inter_time=0;iter_inter_time<N;iter_inter_time++){
			int i=int(FRANDOM*N);
			double diffm = 0.0;
			expbeta=exp(beta);
			expmb=exp(-beta);
			if(graph_neis[i].size()<LARGE_DEGREE) diffm=mod_bp_iter_update_psi(i,dumping_rate);
			else diffm=mod_bp_iter_update_psi_large_degree(i,dumping_rate);
			if(diffm > maxdiffm) maxdiffm=diffm;
		}
		if(vflag >= 2) cout<<maxdiffm<<" "<<flush;
		else if(vflag == 1) cout<<"."<<flush;
		if(iter_time<10) continue;
		if(maxdiffm < bp_err ){
			if(vflag >= 1) cout<<" bp-:) iter_time="<<iter_time<<" "<<endl;
			if(vflag >= 10) show_marginals(vflag);
			else if(vflag >= 3) show_marginals(10);
			bp_last_conv_time=iter_time;
			bp_last_diff=maxdiffm;
			return iter_time;
		}
	}
	bp_last_conv_time=max_iter_time;
	if(vflag >= 1) cout<<" bp-:( iter_time="<<max_iter_time<<" "<<endl;
	if(vflag >= 3) show_marginals(10);
	return max_iter_time;
}
//}}}
//{{{void modbp:: bp_set_true_message(int i)
void modbp:: bp_set_true_message(int i)
{
	for(int q=0;q<Q;q++) {
		if(q==conf_true[i]) {
			if(beta>999) real_psi[i][q] = 0;
			else real_psi[i][q] = 1.0;
		}else{
			if(beta>999) real_psi[i][q]=-1.0;
			else real_psi[i][q]=0.0;
		}
	}
	for (int idxij=0;idxij<graph_neis[i].size();idxij++) {
		int j=graph_neis[i][idxij];
		int idxji=graph_neis_inv[i][idxij];
		for(int q=0;q<Q;q++) {
			if(q==conf_true[i]){
				if(beta>999) psi[j][idxji][q] = 0.0;
				else psi[j][idxji][q] = 1.0;
			}else {
				if(beta>999) psi[j][idxji][q] = -1.0;
				else psi[j][idxji][q] = 0.0;
			}
		}
	}
}
//}}}
//{{{void modbp:: bp_set_random_message(int i)
void modbp:: bp_set_random_message(int i)
{
	double norm=0.0;
	for(int q=0;q<Q;q++) {
		real_psi[i][q] = FRANDOM;
		norm += real_psi[i][q];
	}
	for(int q=0;q<Q;q++) real_psi[i][q] /= norm;
	if(beta>999){
		for(int q=0;q<Q;q++) real_psi[i][q] = log(real_psi[i][q]);
	}
	for (int idxij=0;idxij<graph_neis[i].size();idxij++) {
		int j=graph_neis[i][idxij];
		int idxji=graph_neis_inv[i][idxij];
		norm=0.0;
		for(int q=0;q<Q;q++) {
			psi[j][idxji][q] = FRANDOM;
			norm += psi[j][idxji][q];
		}
		for(int q=0;q<Q;q++) psi[j][idxji][q] /= norm;
		if(beta>999){
			for(int q=0;q<Q;q++) psi[j][idxji][q]=log(psi[j][idxji][q]);
			//for(int q=0;q<Q;q++) psi[j][idxji][q]=psi[j][idxji][q]*(-10);
		}
	}
}
//}}}
//{{{bool modbp:: check_factorized_solution(double err)
bool modbp:: check_factorized_solution(double err)
{
	err=0.0001;
	//cout<<"err="<<err<<endl;
	int count=0;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++){
			if(fabs(real_psi[i][q]-1.0/Q)<err) count++;
		}
	}
	//cout<<"count of factorized "<<count<<endl;
	if(count >= 0.5*N*Q) return true;
	else return false;
}
//}}}

//Inference, retrieval, select and hierarchical clusteirng part
//{{{ void modbp::do_inference(double conv_crit, int time_conv, double dumping_rate)
void modbp::do_inference(double conv_crit, int time_conv, double dumping_rate)
{
	int niter=0;
	bp_allocate();
	niter=bp_converge(conv_crit,time_conv, 1, dumping_rate);
	output_f_ovl(1);
	compute_argmax_conf();	
	compute_cuts();
	cout<<"retrieval_modularity="<<modularity<<endl;
}
//}}}
//{{{bool modbp::retrieval(double bp_conv_crit, int time_conv, double dumping_rate)
bool modbp::retrieval(double bp_conv_crit, int time_conv, double dumping_rate,int myvflag)
{
	//find retrieval state for given q
	bp_allocate();
	double beta_max=10.0,beta_min=0.0;
	double beta_a=beta_min,beta_b=beta_max;
	double c=2.0*M/N;
	if(c<1.00001) beta=0.01;
	else beta=log((Q-1.0+sqrt(c))/(sqrt(c)-1.0)); //initial beta value
	//cout<<"initial beta="<<beta<<" Q="<<Q<<" c="<<c<<" N="<<N<<" M="<<M<<endl;
	double tovl=1.0/Q+0.005;
	retrieval_state=false;
	retrieval_modularity=0;
	while(true){
		int niter=bp_converge(bp_conv_crit,time_conv, 1, dumping_rate);
		double maxovl=compute_overlap_marg();
		compute_cuts();
		if(myvflag>0) cout<<"# beta="<<beta<<" conv_time="<<niter<<" maxovl="<<maxovl<<" "<<" modularity="<<modularity<<endl;
		//cout<<"# beta="<<beta<<" conv_time="<<niter<<" maxovl="<<maxovl<<" "<<" modularity="<<modularity<<endl;
		if(bp_last_conv_time<time_conv-1 && maxovl>=tovl){ // BP has converged and found non-trivial solution
			retrieval_state=true;
			retrieval_modularity=modularity;
			//retrieval state found
			cout<<"# retrieval state found: Modularity="<<modularity<<endl;
			//output_f_ovl(1);
			if( !opt) {
				return true;
			}
			//optimize modularity
			double tmp=beta;
			beta = (beta+beta_b)/2.0;
			beta_a=tmp;
		}else if(bp_last_conv_time<time_conv-1 && maxovl<=tovl){// BP has converged and found the factorized solution
			double tmp=beta;
			beta = (beta+beta_b)/2.0;
			beta_a=tmp;
		}else if(bp_last_conv_time>=time_conv-1){ // BP does not converge
			double tmp=beta;
			beta = (beta+beta_a)/2.0;
			beta_b=tmp;
		}else{
			cout<<"seems a problem occured, please write to pan (pan@santafe.edu) to report the bug, thanks!"<<endl;
			if(myvflag>0) cout<<"# no retrieval state could be found"<<endl;
			cout<<"bp_last_conv_time"<<bp_last_conv_time<<" maxovl="<<maxovl<<endl;
			cout<<"beta="<<beta<<" N="<<N<<" M='"<<M<<" maxd="<<graph_max_degree<<endl;
			//exit(0);
			return false;
		}
		if(fabs(beta_a-beta_b)<0.01){
			if(retrieval_state){
				if(myvflag>0) {
					cout<<"# retrieval state found: Modularity="<<retrieval_modularity<<endl;
				}
				return true;
			}else{
				if(myvflag>0) cout<<"# no retrieval state could be found"<<endl;
				return false;
			}
		}
	}
}
//}}}
//{{{bool modbp::sg(double bp_conv_crit, int time_conv, double dumping_rate)
bool modbp::sg(double bp_conv_crit, int time_conv, double dumping_rate,int myvflag)
{
	//find retrieval state for given q
	bp_allocate();
	double beta_max=20.0,beta_min=0.0;
	double beta_a=beta_min,beta_b=beta_max;
	double c=2.0*M/N;
	beta=log((Q-1.0+sqrt(c))/(sqrt(c)-1.0)); //initial beta value
	double tovl=1.0/Q+0.005;
	retrieval_state=false;
	retrieval_modularity=0;
	while(true){
		int niter=bp_converge(bp_conv_crit,time_conv, 1, dumping_rate);
		double maxovl=compute_overlap_marg();
		if(myvflag>0) cout<<"# beta="<<beta<<" conv_time="<<niter<<" maxovl="<<maxovl<<" "<<" modularity="<<modularity<<" ";
		if(bp_last_conv_time<time_conv-1 && maxovl>=tovl){
			//retrieval state found
			//cout<<"# retrieval state found: Modularity="<<modularity<<endl;
			double tmp=beta;
			beta = (beta+beta_b)/2.0;
			beta_a=tmp;
			cout<<"retrieval"<<endl;
		}else if(bp_last_conv_time<time_conv-1 && maxovl<=tovl){//factorized solution
			double tmp=beta;
			beta = (beta+beta_b)/2.0;
			beta_a=tmp;
			cout<<"trivial"<<endl;
		}else if(bp_last_conv_time>=time_conv-1){
			double tmp=beta;
			beta = (beta+beta_a)/2.0;
			beta_b=tmp;
			cout<<"no_converge"<<endl;
		}else{
			cout<<"seems a problem occured"<<endl;
			exit(0);
		}
		if(fabs(beta_a-beta_b)<0.001){
			cout<<"spin glass transition found! beta_sg="<<beta_b<<endl;
			beta_sg=beta_b;
			return true;
		}
		if(beta_a>19.0){
			cout<<"spin glass transition is larger than 19!"<<endl;
			beta_sg=19.0;
			return true;
		}
	}
}
//}}}
//{{{int modbp:: select(double bp_conv_crit, int time_conv, double dumping_rate)
int modbp:: select(double bp_conv_crit, int time_conv, double dumping_rate,int myvflag)
{
	//int Q_max=10;
	//cout<<"Q_max="<<Q_max<<endl;
	if(graph_edges.size()==0){
		cout<<"No edges! N="<<N<<"M="<<M<<endl;
		Q=N;
		groups_infer.clear();
		for(int i=0;i<N;i++) conf_infer[i]=i;
		return Q;
	}
	double last_mod=-0.1;
	bool stopflag=false;
	vector <int> last_conf_infer;
	vector < vector <int> > last_groups_infer;
	//const double delta5=6.0/N;//0.01;
	//const double delta10=6.0/N;//0.005;
	//const double deltalarge=6.0/N;//0.01;
	const double delta5=0.001;
	const double delta10=0.001;
	const double deltalarge=0.001;
	for(Q=2;Q<Q_max;Q++){
		//for(int q=0;q<Q;q++) na[q]=1.0/Q*N;
		bool find_rs = retrieval(bp_conv_crit, time_conv, dumping_rate,0);
		if(myvflag>0) {
			cout<<"### Q="<<Q<<" modularity="<<retrieval_modularity<<endl;
			output_f_ovl(1);
		}
		if(Q<=5){
			if(retrieval_modularity-last_mod<delta5){//to deal with fluctuations of modularity values
				stopflag=true;
			}
		}else if(Q<=10){
			if(retrieval_modularity-last_mod<delta10){//to deal with fluctuations of modularity values
				stopflag=true;
			}
		}else{
			if(retrieval_modularity-last_mod<deltalarge){//to deal with fluctuations of modularity values
				stopflag=true;
			}
		}	
		if(stopflag) {
			retrieval_modularity=last_mod;
			Q--;
			conf_infer.clear(); conf_infer=last_conf_infer;
			groups_infer.clear(); groups_infer=last_groups_infer;
			break;
		}else {
			last_mod=retrieval_modularity;
			last_conf_infer.clear(); last_conf_infer=conf_infer;
			last_groups_infer.clear(); last_groups_infer=groups_infer;
		}
	}
	if(retrieval_modularity<0.01){
		cout<<"# Random graph found"<<endl;
		Q=0;
		groups_infer.clear();
		for(int i=0;i<N;i++) conf_infer[i]=0;
	}else{
		cout<<Q<<" groups selected, retrieval_modularity="<<retrieval_modularity<<endl;
	}
	return Q;
}
//}}}
//{{{void modbp:: hiera(double bp_conv_crit, int time_conv, double dumping_rate, int myflag)
void modbp:: hiera(double bp_conv_crit, int time_conv, double dumping_rate, int myflag)
{
	queue <tnode *> to_select;
	//root node, the top one
	dendo.resize(1);
	dendo[0].resize(1);//root node
	tnode *root=new tnode;
	dendo[0][0] = root;
	root->level=0;
	root->nodes.resize(N);
	for(int i=0;i<N;i++) root->nodes[i]=i;
	root->father=NULL;
	root->father_idx=-1;
	root->idx=0;
	root->graph_neis=graph_neis;
	root->graph_edges=graph_edges;
	to_select.push(root);
	

	while( !to_select.empty() ){
		tnode *leaf=to_select.front();
		to_select.pop();
		if(leaf->graph_neis.size()==0) {
			continue; // to avoid empty graphs
		}
		copy_struct(leaf);
		select(bp_conv_crit,time_conv,dumping_rate,0); //model selection on this leaf
		leaf->groups_infer=groups_infer;
		leaf->conf_infer=conf_infer;
		leaf->build_children(dendo,to_select);
	}
	copy_struct(root);
}
//}}}
//{{{void modbp:: hiera2(double bp_conv_crit, int time_conv, double dumping_rate, int myflag)
void modbp:: hiera2(double bp_conv_crit, int time_conv, double dumping_rate, int myflag)
{
	queue <tnode *> to_select;
	//root node, the top one
	Q=2;
	dendo.resize(1);
	dendo[0].resize(1);//root node
	tnode *root=new tnode;
	dendo[0][0] = root;
	root->level=0;
	root->nodes.resize(N);
	for(int i=0;i<N;i++) root->nodes[i]=i;
	root->father=NULL;
	root->father_idx=-1;
	root->graph_neis=graph_neis;
	root->graph_edges=graph_edges;
	to_select.push(root);

	while( !to_select.empty() ){
		tnode *leaf=to_select.front();
		to_select.pop();
		copy_struct(leaf);
		//select(bp_conv_crit,time_conv,dumping_rate,0); //model selection on this leaf
		bool find_rs = retrieval(bp_conv_crit, time_conv, dumping_rate,0);
		if(retrieval_modularity<0.01){
			cout<<"# Random graph found"<<endl;
			groups_infer.clear();
			for(int i=0;i<N;i++) conf_infer[i]=0;
		}

		leaf->groups_infer=groups_infer;
		leaf->conf_infer=conf_infer;
		leaf->build_children(dendo,to_select);
	}
//	copy_struct(root);
//	cout<<"N="<<graph_neis.size()<<endl;
}
//}}}
//{{{void tnode:: build_children(vector < vector < tnode *> > &dendo, queue < tnode * > &to_select)
void tnode:: build_children(vector < vector < tnode *> > &dendo, queue < tnode * > &to_select)
{
	if(groups_infer.size()==1 ||groups_infer.size()==0){ // random network or single group, anyway, no need to build children.
		return;
	}
	//produce children, when data structure of root and conf_infer are set:
	this->nodes_assign.resize(this->graph_neis.size());
	this->children.resize(groups_infer.size());
	this->children_idx.resize(groups_infer.size());
	if(dendo.size()<=this->level+1){
		vector <tnode *> ltmp;
		dendo.push_back(ltmp);
	}
	for(int q=0;q<groups_infer.size();q++){
		tnode *noa=new tnode;
		noa->father= this;
		noa->father_idx=this->idx;
		noa->level = (noa->father->level)+1;
		noa->idx = dendo[noa->level].size();
		dendo[noa->level].push_back(noa);
		this->children_idx[q] = dendo[noa->level].size();
		noa->nodes.resize(groups_infer[q].size());
		noa->graph_neis.resize(groups_infer[q].size());
		for(int idx=0;idx<groups_infer[q].size();idx++){
			int i=this->nodes[this->groups_infer[q][idx]];
			noa->nodes[idx]=i;// original index
			this->nodes_assign[groups_infer[q][idx]]=idx;
		}
		this->children[q] = noa;
		to_select.push(noa);
	}
	for(int idxa=0;idxa<this->graph_edges.size();idxa++){
		int fi=this->graph_edges[idxa][0];
		int fj=this->graph_edges[idxa][1];
		//cout<<fi<<" "<<fj<<endl;
		if(this->conf_infer[fi] == this->conf_infer[fj]){
			tnode *noa=this->children[conf_infer[fi]];
			int i=this->nodes_assign[fi];
			int j=nodes_assign[fj];
			noa->graph_neis[i].push_back(j);
			noa->graph_neis[j].push_back(i);
			vector <int> tmp;
			tmp.push_back(i); tmp.push_back(j);
			noa->graph_edges.push_back(tmp);
		}
	}
}
//}}}
//{{{void modbp::print_dendo(string fconf_infer)
void modbp::print_dendo(string fconf_infer)
{
	Q=0;
//	cout<<"-------------------------------"<<endl;
	//cout<<"------------------------------ Dendrogram: ---------------------------"<<endl;
	cout<<"----------------------------------------------------------------------"<<endl;
	cout<<"Dendogram has "<<dendo.size()<<" layers"<<endl;
	conf_infer.resize(N);
	for(int l=0;l<dendo.size();l++){
		cout<<"layer "<<l<<" :";
		for(int i=0;i<dendo[l].size();i++){
			cout<<dendo[l][i]->nodes.size();
			//for(int j=0;j<dendo[l][i]->nodes.size();j++){
			//	cout<<dendo[l][i]->nodes[j]<<" ";
			//	cout<<endl;
			//}
			if(dendo[l][i]->groups_infer.size()<=1) {
				for(int j=0;j<dendo[l][i]->nodes.size();j++) conf_infer[dendo[l][i]->nodes[j]]=Q;
				cout<<"*";
				Q++;
			}else{
				cout<<"("<<dendo[l][i]->children.size()<<") ";
			}

			//cout<<"("<<2.0*dendo[l][i]->graph_edges.size()/dendo[l][i]->graph_neis.size()/dendo[l][i]->graph_neis.size()<<") ";
			//cout<<"("<<dendo[l][i]->children.size()<<") ";
			cout<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	//cout<<"--------------------At different levels of dendrogram: ---------------"<<endl;
	cout<<"----------------------------------------------------------------------"<<endl;
	cout<<"Index of father at last level "<<endl;
	cout<<"Number in the bracket is the final label"<<endl;
//	cout<<"------------------------------ Dendrogram: ---------------------------"<<endl;
	ofstream fout;
	if(!fconf_infer.empty()) {
		string tmpstr=fconf_infer+string(".tree");
		fout.open(tmpstr.c_str());
	}
	vector < vector <int> > fixed_groups;
	for(int l=1;l<dendo.size();l++){
		cout<<"layer "<<l<<": ";
		Q=fixed_groups.size();
//		cout<<endl;
		for(int q=0;q<Q;q++){
			for(int i=0;i<fixed_groups[q].size();i++){
				conf_infer[fixed_groups[q][i]]=q;
			}
		}
		for(int i=0;i<dendo[l].size();i++){
			cout<<dendo[l][i]->father_idx;
			if(!fconf_infer.empty()) fout<<dendo[l][i]->father_idx;
			for(int j=0;j<dendo[l][i]->nodes.size();j++) conf_infer[dendo[l][i]->nodes[j]]=Q;
			Q++;
			if(dendo[l][i]->groups_infer.size()<=1) {//one leaf of the tree
				cout<<"("<<fixed_groups.size()<<") ";
				if(!fconf_infer.empty()) fout<<"("<<fixed_groups.size()<<") ";
				vector <int> fixed_vars;
				for(int j=0;j<dendo[l][i]->nodes.size();j++) fixed_vars.push_back(dendo[l][i]->nodes[j]);
				fixed_groups.push_back(fixed_vars);
			}else {
				cout<<" ";
				if(!fconf_infer.empty()) fout<<" ";
			}
		}
		form_groups_infer();
		compute_cuts();
//		cout<<"Q="<<Q<<" N="<<graph_neis.size()<<" M="<<graph_edges.size()<<" fixed groups="<<fixed_groups.size()<<" modularity="<<modularity<<endl;
//		cout<<endl<<"Q="<<Q<<" N="<<graph_neis.size()<<" M="<<graph_edges.size()<<" modularity="<<modularity<<endl;
		cout<<" \tModularity="<<modularity<<endl;
		if(!fconf_infer.empty()) {
			fout<<endl;
			char tmp[128];
			sprintf(tmp,".%d",l);
			mod_record_conf_infer(fconf_infer+string(tmp));

		}
	}
	cout<<"----------------------------------------------------------------------"<<endl;
}
//}}}
//{{{void modbp::copy_struct(tnode *leaf)
void modbp::copy_struct(tnode *leaf)
{
	graph_neis.clear();graph_neis=leaf->graph_neis;
	graph_edges.clear();graph_edges=leaf->graph_edges;
	N=graph_neis.size();
	M=graph_edges.size();
	graph_di.resize(N);
	for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size();
	graph_build_neis_inv();//build neighbor matrix graph_neis[][] and graph_neis_inv[][] from edge matrix graph_edges.
}
//}}}

//Output part
//{{{void modbp:: output_f_ovl(int mode)
void modbp:: output_f_ovl(int mode)
{
	double f=0.0;
	f=mod_bp_compute_f_large_degree();
	cout<<"free_energy="<<f<<deli;
	if(groups_true.size() > 0 && groups_true.size()==Q) cout<<"fraction_reconstructed="<<compute_overlap_fraction()<<endl;
	else cout<<endl;
}
//}}}
//{{{void modbp:: output_marginals(string fname)
void modbp:: output_marginals(string fname)
{
	ofstream fout(fname.c_str());
	assert(fout.good() && "I can not open file to write marginals.");
	double f=0.0;
	f=mod_bp_compute_f();
	fout<<"f="<<f<<deli<<"argmax_ovl="<<compute_overlap_marg()<<deli;
	if(groups_true.size() > 0) fout<<"overlap="<<compute_overlap()<<deli<<"fraction="<<compute_overlap_fraction()<<endl;
	else fout<<endl;
	compute_argmax_conf();
	fout<<"argmax_configuration:"<<endl;
	for(int i=0;i<N;i++) fout<<conf_infer[i]<<" ";
	fout<<endl;
	fout<<endl;

	fout<<"marginals:"<<endl;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++) fout<<real_psi[i][q]<<" ";
		fout<<endl;
	}
	fout<<endl;
	fout<<"argmax_marginals:"<<endl;
	for(int i=0;i<N;i++) fout<<argmax_marginals[i]<<" ";
	fout<<endl;
	cout<<"marginals, argmax configuration, argmax marginals are written into "<<fname<<"."<<endl;
}
//}}}
//{{{void modbp::show_marginals(int num)
void modbp::show_marginals(int num)
{
	cout<<"Marginals from node 0 to node "<<num<<":"<<endl;
	for(int i=0;i<num;i++){
		double lpsi=-1.0;
		int lq=0;
		for(int q=0;q<Q;q++) {
			cout<<real_psi[i][q]<<" "<<flush;
			if(real_psi[i][q]>lpsi) {
				lpsi=real_psi[i][q];
				lq=q;
			}
		}
		cout<<lq<<endl;
//		cout<<endl;
	}
	cout<<endl;
//	for(int i=0;i<N;i++){
//		cout<<conf_infer[i]<<" "<<flush;
//	}
//	cout<<endl;
//	cout<<"similarity of groups:"<<endl;
//	for(int q=0;q<Q;q++){
//		for(int t=q+1;t<Q;t++){
//			double sim=0.0;
//			for(int i=0;i<N;i++) sim += fabs(real_psi[i][q]-real_psi[i][t]);
//			sim /= N;
//			cout<<q<<t<<"="<<sim<<" ";
//		}
//		cout<<endl;
//	}

}
//}}}

//permutation
//{{{ void modbp::init_perms()
void modbp::init_perms()
{
	perms.clear();
	vector <int> myperm;
	for(int i=0;i<Q;i++) myperm.push_back(i);
	sort(myperm.begin(),myperm.end());
	if(Q>Q_PERMU){
		int tt_max=2000;
		for(int tt=0;tt<tt_max;tt++){
			perms.push_back(myperm);
			next_permutation(myperm.begin(),myperm.end());
		}
	}else{
		do perms.push_back(myperm);
		while(next_permutation(myperm.begin(),myperm.end()));
	}
}
//}}}

//spectral clustering part
//{{{string get_std_from_cmd(string cmd)
string get_std_from_cmd(const char* cmd)
{
	string data;
	FILE *stream;
	char buffer[1024];
	stream = popen(cmd, "r");
	while ( fgets(buffer, 1024, stream) != NULL )
	data.append(buffer);
	pclose(stream);
	return data;
}
//}}}
//{{{vector <string> strsplit(const string& str, const string& delimiters = " ")
vector <string> strsplit(const string& str, const string& delimiters = " ")
{
	vector <string> tokens;
	// Skip delimiters at beginning.
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos){
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
	return tokens;
}
//}}}
//{{{long get_cpu_time(void)
long get_cpu_time(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ( ru.ru_utime.tv_sec*1000 +
	     ru.ru_utime.tv_usec/1000+
	     ru.ru_stime.tv_sec*1000 +
	     ru.ru_stime.tv_usec/1000 );
}
//}}}

