# 11420PHYS — Computational Many-Body Physics  
**物理專題—多體物理計算**

Department of Physics, National Tsing Hua University  
Spring 2026

Instructor: **Professor Ian McCulloch(麥伊安)**  ian@phys.nthu.edu.tw
* Consulting hours: Thursdays 1200 - 1300 in R715

Teaching Assistant: **Che-Chia Hsu (許哲嘉)** charles123756@gapp.nthu.edu.tw
* Consulting hours: Tuesdays 1110 - 1200 in R708

---

## Assignment Grading

- [Grading Criteria (English)](grading.md)
- [評分標準 (中文)](grading-zh.md)

---

## Assignments

* Assignment 1: Exact Diagonalization https://classroom.github.com/a/eZv96eFT due: 2026-03-20 13:20

* Assignment 2: Infinite DMRG https://classroom.github.com/a/-MLdj3q7 due: 2026-04-10 13:20

* Assignment 3: Finite-System DMRG for the Spin-1 Chain https://classroom.github.com/a/BZmkyJUq due 2026-05-08 13:20

* Assignment 4: Finite-System real-time evolution and spectral functions https://classroom.github.com/a/cH5hz_AW due 2026-05-08 13:20

---

## Course description

This course introduces computational methods for studying quantum many-body systems, with an emphasis on **hands-on numerical modeling using Python**.

Students will learn how to

- construct many-body Hamiltonians from local operators
- diagonalize quantum lattice models
- compute physical observables
- visualize results and interpret numerical experiments

---

## Course structure

Each week typically includes

- short lecture introducing a physical or numerical concept
- live coding demonstration
- in-class programming exercises
- discussion of results

Assignments will involve small computational projects using Python.

---

## Repository contents

This repository contains

```

lecture-notes/       slides and lecture material
examples/            small example programs used in class
environment.yml      Python environment specification

```

Assignments are handled through the **GitHub Classroom**.

---

## Software requirements

Students should bring a **laptop computer** to class.

The course uses Python together with the following scientific libraries:

- NumPy
- SciPy
- Matplotlib

To simplify installation and ensure a consistent environment we use **Conda**.

---

## Installing Conda

We recommend installing **Miniforge**, a minimal Conda distribution.

Download the installer for your system:

https://github.com/conda-forge/miniforge/releases/latest

Choose the appropriate installer:

| System | Installer |
|------|------|
Linux | `Miniforge3-Linux-x86_64.sh` |
Mac (Intel) | `Miniforge3-MacOSX-x86_64.sh` |
Mac (Apple Silicon) | `Miniforge3-MacOSX-arm64.sh` |
Windows | `Miniforge3-Windows-x86_64.exe` |

### Linux / macOS installation

Open a terminal and run

```

bash Miniforge3-*.sh

```

Accept the default options.

Restart the terminal after installation.

### Windows installation

Run the `.exe` installer and follow the instructions.

After installation open the **Miniforge Prompt**.

---

## Creating the course Python environment

After cloning this repository, create the course environment:

```

conda env create -f environment.yml

```

Activate it with

```

conda activate manybody

```

You should see

```

(manybody)

```

at the start of your terminal prompt.

---
