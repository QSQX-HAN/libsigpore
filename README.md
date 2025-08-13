libsigpore
==============

Library holding utility functions to perform single molecule heterogeneity analysis


Description
===========

Sigpore ingests Oxford Nanopore's Direct RNA sequencing data to automatically detect synthetic chemical probes used to identify RNA nucleotides that are not paired in a secondary structure fashion. These unstructured bases are biochemically reactive to NAIN3, a SHAPE-derivative that is routinely used in the lab for structure probing. This library supports the downstream analysis of Sigpore's processed results seen in notebooks.

Software Pre-requisites
------------------------

  1. Vienna RNAfold (with python API interface)


Installation Requirements
--------------------------

  1. altair == 4.1.0
  2. bayespy == 0.5.22
  3. CairoSVG == 2.7.0
  4. python-louvain == 0.15
  5. fancyimpute == 0.7.0
  6. forgi == 2.0.2
  7. h5py == 3.8.0
  8. igraph == 0.9.9
  9. intervaltree == 3.1.0
  10. leidenalg == 0.8.9
  11. matplotlib >= 3.5.1
  12. networkx == 2.6.2
  13. numba == 0.55.1
  14. numpy <= 1.21.5
  15. pandas >= 1.1.5
  16. pyfaidx >= 0.6.1
  17. qstest == 1.0.3
  18. ruptures >= 1.1.6
  19. scipy >= 1.5.4
  20. seaborn >= 0.11.2
  21. scikit-learn >= 0.24.2
  22. tslearn >= 0.5.2
  23. umap-learn == 0.5.1


This will be automatically done during `pip install`

TL;DR
------
~~~
conda create -n sm-PORE-cupine python=3.9
conda activate sm-PORE-cupine
conda install -c bioconda viennarna=2.4.18
git clone https://github.com/YueLab-GIS-ASTAR/libsigpore
cd libsigpore
pip install -e .
~~~

