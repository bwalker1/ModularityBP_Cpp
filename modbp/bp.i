%module bp

%{
#include "src_cpp/bp.h"
%}
%include typemaps.i
%apply double *OUTPUT {double& s}

%include "src_cpp/bp.h"
