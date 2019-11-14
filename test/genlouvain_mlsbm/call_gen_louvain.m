function call_gen_louvain(input_file,output_file,gamma,coupling)
    load(input_file);
    A=sparse(A);
    A=max(A,A'); %symmetrize these
    C=max(C,C');
    %P should be in the file already
    P=P;


    B=A-gamma*P+coupling*C;
    rng('shuffle');
    if exist('S0','var')
          tic,[S,Q]=genlouvain(B,20000,0,1,'moverandw',S0);toc;
    else
          tic,[S,Q]=genlouvain(B,20000,0,1,'moverandw');toc;
    end

    save(output_file,'S');
   
    
