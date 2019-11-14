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




export MATLABPATH="${homedir}/test/genlouvain_mlsbm/"
export MATLABPATH="${MATLABPATH}:${homedir}/test/genlouvain_mlsbm/GenLouvain-master"

#X="addpath $matlab_func_file;"$'\n'

#function call gets written to shell script
X="call_gen_louvain('${1}','${2}',${3},${4})"
#X="which(call_gen_louvain)"

dir=${1%/*} #get directory of input file 
base=${1##*/}
echo ${X} > $dir/matlab_command_$base.m
# cat matlab_command_${2}.m
# chown 0777 matlab_command_${2}.m
${matlab_exec} -nojvm -nodisplay -nosplash -nodesktop < $dir/matlab_command_$base.m
rm $dir/matlab_command_$base.m