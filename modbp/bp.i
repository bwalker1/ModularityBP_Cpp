%module bp

%{
#include "src_cpp/bp.h"
%}
%include typemaps.i
%apply double *OUTPUT {double& s}

%include "src_cpp/bp.h"
%include "std_vector.i"
%include "std_pair.i"
%template() std::pair <unsigned long,unsigned long>;
%template(PairVector) std::vector< std::pair < unsigned long, unsigned long > >;

#%template(Line) vector < unsigned long >;
#%template(Array) vector < vector <unsigned long> >