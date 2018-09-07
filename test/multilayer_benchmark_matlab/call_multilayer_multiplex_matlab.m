function call_multilayer_multiplex_matlab(output_file,n_nodes,n_layers,mu,p,ncoms)
    L = MultiplexDependencyMatrix(n_layers,p);
    %L = TemporalDependencyMatrix(n_layers,p);
    [A,S]=DirichletDCSBMBenchmark(n_nodes,n_layers,L,...
    'UpdateSteps',200,'theta',1,'communities',ncoms,'q',1,...
    'exponent',-2,'kmin',3,'kmax',150,'mu',mu,'maxreject',100);
    save(output_file,'S','L','A');
    0
    
