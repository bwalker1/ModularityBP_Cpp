#!/bin/bash

#matlab_exec=/Applications/MATLAB_R2016b.app/bin/matlab
#matlab_exec=/nas/longleaf/apps/matlab/2017a/bin/matlab
module add matlab
matlab_exec=matlab
homedir='/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp'

matlab_func_file="${homeddir}/test/genlouvain_mlsbm/run_gen_louvain.m"
export MATLABPATH="${homedir}/test/genlouvain_mlsbm/"
export MATLABPATH="${MATLABPATH}:${homedir}/test/genlouvain_mlsbm/GenLouvain-master"

#X="addpath $matlab_func_file;"$'\n'

X="call_gen_louvain('${1}','${2}','${3}','${4}')"
dir=${1%/*} #get directory of input file 
base=${1##*/}
#echo ${X}
echo ${X} > $dir/matlab_command_$base.m
# cat matlab_command_${2}.m
# chown 0777 matlab_command_${2}.m
${matlab_exec} -nojvm -nodisplay -nosplash -nodesktop < $dir/matlab_command_$base.m
rm $dir/matlab_command_$base.m