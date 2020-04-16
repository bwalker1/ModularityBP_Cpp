function call_multilayer_multiplex_block_matlab(output_file,n_nodes,n_layers,nblocks,mu,p_in,p_out,ncoms,is_multiplex)
    %L = MultiplexDependencyMatrix(n_layers,p);
    if is_multiplex
        L = HeterogeneousMultiplexDependencyMatrix(n_layers,nblocks ,p_in,p_out);
        [A,S]=DirichletDCSBMBenchmark(n_nodes,n_layers,'r',L,...
    'UpdateSteps',200,'theta',1,'communities',ncoms,'q',1,...
    'exponent',-2,'kmin',3,'kmax',150,'mu',mu,'maxreject',300);
    else
        %p_in here is copying community within block. p_out is copying moving across
        %nc is number of change points (i.e. nblocks -1)
        L = TemporalCPDependencyMatrix(n_layers,p_in,p_out,nblocks-1);
        %L = TemporalDependencyMatrix(n_layers,p_in);
        %adjust DCSBM parameters here
        [A,S]=DirichletDCSBMBenchmark(n_nodes,n_layers,'r',L,...
    'UpdateSteps',200,'theta',1,'communities',ncoms,'q',1,...
    'exponent',-2,'kmin',3,'kmax',30,'mu',mu,'maxreject',300);
    end
    size(L)

    save(output_file,'S','L','A');
    0


