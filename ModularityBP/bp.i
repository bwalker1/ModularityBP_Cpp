%module bp

%{
#include "bp.h"
%}

%include typemaps.i
%apply double *OUTPUT {double& s}

%include "bp.h"
