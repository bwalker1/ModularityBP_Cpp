

ModularityBP Object
****************************************

We have created a single Python object, :mod:`modbp.ModularityBP`

1. :ref:`Store graphs on ModularityBP Object <store_graph>`
2. :ref:`Run modularity belief propagation  <run_bp>`
3. :ref:`Access discovered partitions and marginals  <parts_margs>`
4. :ref:`Compile results across many different runs <assessing_res>`


=================================================
Storing Graphs on ModularityBP Object
=================================================
..  _`store_graph`:
A representation of a network must be supplied at the instantiation of the :mod:`modbp.ModularityBP` object.  Each instance can only be associated with one network.  \
A network can be supplied in several different formats:

- A :mod:`igraph.Graph` object is supplied.  In this case network will be treated as single layer.  This is pass in through the mlgraph parameter::

    rand_g=igraph.Graph.ErdosRenyi(n=100,p=.05)
    modbp_obj=modbp.ModularityBP(mlgraph=rand_g)


- A :mod:`modbp.MultilayerGraph` object can be supplied, also through the mlgraph parameter.  See :ref:`Multilayer Graph<GenerateGraphs.MultilayerGraph>` for more details.
- Finally, one can pass in an array for intralayer_edges, interlayer_edges, and layer_vec, from which :mod:`modbp.ModularityBP` will internally construct a :mod:`modbp.MultilayerGraph`::

    intra_edges=np.array([[0,1],[0,2],[3,4]])
    inter_edges=np.array([[2,3]])
    layer_vec=[0,0,0,1,1]
    modbp_obj=modbp.ModularityBP(intra_edges=intra_edges,
        inter_edges=inter_edges,
        layer_vec=layer_vec)


The graph is stored internally each :mod:`modbp.ModularityBP` as a :mod:`modbp.MultilayerGraph` and is accessible through :mod:`modbp.ModularityBP.graph` variable.

=================================================
Running Multilayer Modularity Belief Propagation
=================================================
..  _`run_bp`:

.. automethod:: modbp.ModularityBP.run_modbp


=================================================
Accessing partitions and marginals
=================================================
..  _`parts_margs`:

=================================================
Assessing Results Across Many Different Runs
=================================================
..  _`assessing_res`:


* :ref:`genindex`
* :ref:`search`

