function call_multilayer_multiplex_block_matlab(output_file,n_nodes,n_layers,nblocks,mu,p_in,p_out,ncoms)
    %L = MultiplexDependencyMatrix(n_layers,p);
    L = HeterogeneousMultiplexDependencyMatrix(n_layers,nblocks ,p_in,p_out);
    size(L)
    [A,S]=DirichletDCSBMBenchmark(n_nodes,n_layers,'r',L,...
    'UpdateSteps',200,'theta',1,'communities',ncoms,'q',1,...
    'exponent',-2,'kmin',3,'kmax',150,'mu',mu,'maxreject',300);
    save(output_file,'S','L','A');
    0


