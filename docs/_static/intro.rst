.. CHAMP documentation master file, created by
   sphinx-quickstart on Tue Jul 11 15:50:43 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Background
************


=================================
Introduction: Belief Propagation
=================================




==================
Modularity
==================




=====================
Multilayer Modularity
=====================

One of the strengths of modularity is that it has been extended in a principled way into a variety of network topologies \
in particular the multilayer context.  The multilayer formulation :cite:`Mucha:2010vk` for modularity incorporates the interlayer \
connectivity of the network in the form of a second adjacency matrix :math:`C_{ij}`

.. math::
    :nowrap:

    \begin{equation}
    Q(\gamma)=\frac{1}{2m}\sum_{i,j}{\left( A_{ij}-\gamma \frac{k_ik_j}{2m} \
    +\omega C_{ij}\right)\delta(c_i,c_j)}
    \end{equation}

Communities in this context group nodes within the layers and across the layers.  The inclusion of the :math:`C_ij` \
boost the modularity for communites that include alot interlayer links.  There is an additional parameter, \
:math:`\omega` that tunes how much weight these interlink ties contribute to the modularity.   \


References
___________

.. bibliography:: biblio.bib
    :style: plain
    :filter: docname in docnames



* :ref:`genindex`
* :ref:`search`

