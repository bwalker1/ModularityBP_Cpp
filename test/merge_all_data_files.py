
import os
import sys
import pandas as pd
import argparse
import re
#Constants
DESCRIPTION="""Script to take a folder of csv files and stack them into a single 
            data frame.  Assumes that each file has the same number of columns 
            that are arranged in the same order, and that the index columns is 
            of no importance
            """


def main(pargs=None):
	#default_dir='/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/modbpdata/LFR_test_data'
    default_dir='/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/modbpdata/LFR_test_data_gamma3_beta2'
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input_dir',dest='input',type=str,
                        default=default_dir)
    parser.add_argument('--prefix',dest='prefix',type=str,default=None)
    parser.add_argument('--output_file',dest='outfilename',type=str,default=None)
    
    if pargs is None:
        args=parser.parse_args()
    else:
        args=parser.parse_args(pargs)
    if args.outfilename is None:
	outfile='merged_all_{:}'.format(args.input.split('/')[-1])
    if args.prefix is not None:
        outfile=args.prefix+"_"+outfile
    else:
        outfile=args.outfile
    print(outfile)
    allfiles=os.listdir(args.input)
    # outdf=pd.DataFrame()
    for i,file in enumerate(allfiles):
        if i%1000==0:
            print("{}/{}".format(i,len(allfiles)))
        if not re.search('\.csv',file):
            continue #only append csv files
        cfile=os.path.join(args.input,file)
        c_df=pd.read_csv(cfile,index_col=0)
        if i==0:
            with open(outfile,'w') as fh:
                c_df.to_csv(fh,header=True)
        else:
            with open(outfile,'a') as fh:
                c_df.to_csv(fh,header=False)
    # print ('writing file to {:}'.format(outfile))
    # outdf.to_csv(outfile)
    return 0

if __name__=='__main__':
    sys.exit(main())
