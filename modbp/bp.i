%module bp

%{
#include "src_cpp/bp.h"
typedef unsigned long index_t;
%}
%include typemaps.i
%apply double *OUTPUT {double& s}

%include "src_cpp/bp.h"
%include "std_vector.i"
%include "std_pair.i"
%template() std::pair<unsigned long,unsigned long>;
%template(PairVector) std::vector< std::pair < unsigned long, unsigned long > >;

%template(IntArray) std::vector<index_t>;
%template() std::vector<double>;
%template(Array) std::vector< std::vector<double> >;