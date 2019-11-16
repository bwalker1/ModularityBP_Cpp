function call_gen_louvain(input_file,output_file,gamma,coupling)
    load(input_file);
    A=sparse(A);
    A=max(A,A'); %symmetrize these
    C=max(C,C');
    %P should be in the file already
    P=P;
    T=T; %number of layers
    %post process
    PP = @(S) postprocess_categorical_multilayer(S,T);


    B=A-gamma*P+coupling*C;
    rng('shuffle');
    tic;[S,Q,n_it]=iterated_genlouvain(B,20000,0,1,'moverandw',[],PP); toc; %use post processing for genlouvain


    save(output_file,'S','Q','n_it');


    
