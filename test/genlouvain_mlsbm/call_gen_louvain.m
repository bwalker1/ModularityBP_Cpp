function call_gen_louvain(input_file,output_file,gamma,coupling)
    load(input_file);
    A=sparse(A);
    A=max(A,A');
    C=max(C,C');
    %P should be in the file already
    %P=P;
    offset=0;
    % we manually setup P here.
    P=zeros(size(A));
    nlayers=size(degs);
    for i=1:nlayers(1),
        k=degs(i,:); %row vector here
        twom=sum(k);
        P(offset+(1:length(k)),offset+(1:length(k)))=k'*k/twom;
        offset=offset+length(k);
    end

    B=A-gamma*P+coupling*C;
    rng('shuffle');
    if exist('S0','var')
          tic,[S,Q]=genlouvain(B,20000,0,1,'moverandw',S0);toc;
    else
          tic,[S,Q]=genlouvain(B,20000,0,1,'moverandw');toc;
    end

    save(output_file,'S');
   
    