# csm

**C**ompetitive **S**earch and inflationary **M**onetary equilibrium

Python codes and (Data) Jupyter Notebook for paper *Inflation, Inequality and Welfare in a Competitive Search Model* by Timothy Kam, Tina Kao and Junsang Lee
* **Accepted and Forthcoming**: *Macroeconomic Dynamics* 

Stationary monetary equilibrium in a competitive search environment with free asset-market participation 

Code solves and simulates stationary equilibrium behavior and non-degenerate distribution of agents

A current version of the paper can be found here

* CSSEGMOD: Class file

* MAIN: Script for running an instance of the model

* SETOPS: Some Python tools for set operations (custom coded)

* SETOUTPATHS: Determine where to save results (.NPZ files) and figures (.PNG and .EPS files)
 * Users need to customize to own computer setting

## Dependencies
 
* ``Python 3.8``

* Code uses ``MPI4Py`` and ``OpenMPI`` for parallel computation.

* *Warning* (This is a publicly known issue.) Current/latest distributions of ``Anaconda`` (``Python 3.9`` and higher) fail to have ``MPI4Py`` library build properly (applies for ``pip``, ``conda`` or hand-compiled from source): It fails to link to the executable ``mpicc`` in ``OpenMPI``. Users are advised to create a separate ``env`` installed with ``python 3.8``.

(c) Copyright 2024-, Timothy Kam.

## Code Example

To run the ``MAIN.PY`` file in normal mode (serial):

> ``python main.py``

or, in IPython:

> ``run main``

To run the MAIN.PY file using OpenMPI:

> ``mpirun -n YourNumberOfCPUs python main.py``

or, if you have chmod MAIN.PY to be BASH Executable you can also invoke as

> ``mpirun -n YourNumberOfCPUs ./main.py``

Same goes for ``CALIBRATE.PY`` or ``COMPARESTEADYSTATES.PY``

## Contributors

* Model conceptualization, theoretical proofs, and solution methods developed and written by T. Kam

* Model conceptualization and calibration tasks by J. Lee

* Manuscript re-write, proof checks, and results re-write by T. Kao and T. Kam

## License

The GNU GPL v.3.0 terms apply everywhere to these codes and subsequent uses and modifications of them (see LICENCE). Please cite this original source in all your work (i.e., your own paper and source codes) as:

* Paper: T. Kam, T. Kao and J. Lee (2024): *Inflation, Inequality and Welfare in a Competitive Search Model*

* Source: https://github.com/phantomachine/csm/

