#!/bin/bash

curwd=`pwd`
if [[ $curwd == "*longleaf*" ]];then

    module add matlab
    matlab_exec=matlab
    homedir='/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp'
else
    #for local testing
    matlab_exec=/Applications/MATLAB_R2016b.app/bin/matlab
    homedir='/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp'
fi

export MATLABPATH="${homedir}/test/multilayer_benchmark_matlab/"
export MATLABPATH="${MATLABPATH}:${homedir}/test/multilayer_benchmark_matlab/MultilayerGM-MATLAB-master"
#export MATLABPATH="${MATLABPATH}:${homedir}/test/multilayer_benchmark_matlab/MultilayerBenchmark-master/OptionStruct"

#X="addpath $matlab_func_file;"$'\n'

X="call_multilayer_multiplex_block_matlab('${1}',${2},${3},${4},${5},${6},${7},${8})"
dir=${1%/*} #get directory of input file 
base=${1##*/}
#echo ${X}
echo ${X} > $dir/matlab_command_$base.m
${matlab_exec} -nojvm -nodisplay -nosplash -nodesktop < $dir/matlab_command_$base.m
rm $dir/matlab_command_$base.m