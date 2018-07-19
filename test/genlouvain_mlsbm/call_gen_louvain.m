function call_gen_louvain(input_file,output_file,gamma,coupling)
    load(input_file);
    A=sparse(A);
    A=A+A';
    C=C+C';
    sum(sum(A))
    k=sum(A)';
    twom=sum(k);
    P=k*k'/twom;
    B=A-gamma*P+coupling*C;
    tic,[S,Q]=genlouvain(B,10000,0);toc;
    save(output_file,'S');
    0
    