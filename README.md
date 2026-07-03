## Software requirements

The program requires the following scientific libraries:

- NumPy
- SciPy
- Matplotlib

To simplify installation and ensure a consistent environment we use **Conda**.

## Creating the course Python environment

After cloning this repository, create the environment:

```

conda env create -f environment.yml

```

Activate it with

```

conda activate KitaevDensityMatrix

```

After the environment setting is done, we can test the functions of the module ``SpinModelDM.py`` in the JupyterNotebook ``KitaevDM.ipynb``.

---

# Short Intro

## Lattice Geometry of the Finite-Size System

In this program, we consider a Kitaev model with a periodic boundary condition. The topological sector the ground state lies in is the one with the expectation value of both Wilson loops being 1. The lattice geometry is illustrated below, in which the honeycomb lattice has been reshaped to an equivalent brick-wall geometry. The red dots mark the identical lattice sites under such a periodic boundary condition.

![lattice-geometry](notes/correlation-in-finite-size-system/figures/minimal-system_brick-wall_diagonal.svg)