# Echo Chamber Formation Sharpened by Priority Users
The codes of this paper are implemented in the DOCE (Dynamical Opinion Clusters Exploration Suite) library, which is not yet available. So, we provide a previous version to reproduce our study in this GitHub repository.

Once our library is available, we will replace the codes here.

# DOCES
DOCES is an experimental Python library for simulating opinion dynamics in complex adaptive networks. It is implemented in C for performance reasons.

# INSTALL

It requires Python headers and a C11-compatible compiler, such as gcc or clang. To install it, run the `setup.sh` script, which works for Linux and MacOS. If you are using Windows, please use the setup file in `doces/setup.py` directly.

# Code to reproduce the paper

The Python codes to reproduce the paper can be found in the `figures' folder, where each file reproduces the result of one figure of the paper.

There is no code for Fig. 3 because, in this case, we did not use any Python code. The visualization was done using the visualizer implemented by [Silva et al. 2016](https://doi.org/10.1016/j.joi.2016.03.008.715).
