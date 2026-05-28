# Solving the Schrödinger Equation: Crank-Nicolson vs Fourier Neural Operators

Numerical solution of the time-dependent Schrödinger equation comparing two paradigms: the classical **Crank-Nicolson** finite difference method and **Fourier Neural Operators (FNO)**, a neural architecture that learns the solution operator directly in Fourier space.

---

## Overview

The time-dependent Schrödinger equation describes the quantum dynamics of a particle through a wave function $\psi \in L^2$. While Crank-Nicolson solves the equation step by step with theoretical guarantees of norm preservation, FNO learns the solution operator $G^\dagger: \psi_0 \mapsto \psi(\cdot, t)$ from data, enabling fast inference over large families of initial conditions.

This project implements both approaches for 1D quantum systems with Gaussian wave packets as initial conditions and provides a computational efficiency comparison across varying numbers of initial conditions.

---

## Features

- **Crank-Nicolson solver** with LU factorization and unconditional stability
- **FNO implementation** trained on Crank-Nicolson-generated solutions
- Norm preservation analysis ($\|\psi\|^2 \approx 1$) for both methods
- Computational cost comparison for $N = 1$ to $N = 3000$ initial conditions
- Spacetime probability density visualizations $|\psi(x,t)|^2$
- GPU-accelerated training via Google Colab

---

## Results

| $N$ | CN total (s) | FNO total (s) | Speedup |
|----:|-------------:|--------------:|--------:|
| 1 | 0.018 | 0.132 | 0.1× |
| 10 | 0.141 | 2.974 | 0.0× |
| 50 | 1.154 | 2.739 | 0.4× |
| 100 | 1.552 | 3.016 | 0.5× |
| 500 | 10.280 | 0.720 | 14.3× |
| 1000 | 15.991 | 1.802 | 8.9× |
| 3000 | 48.805 | 5.618 | 8.7× |

FNO achieves up to **14.3× speedup** over Crank-Nicolson for large batches of initial conditions, at the cost of ~8.9% relative L2 error and a norm deviation of order $10^{-2}$.

---

## Training Environment

The FNO was trained on **Google Colaboratory** with GPU backend (Python 3, Google Compute Engine):

- System RAM: 12.7 GB
- GPU RAM: 15.0 GB
- Disk: 112.6 GB

---

## Usage

Open `notebook.ipynb` directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Or clone and run locally:

```bash
git clone https://github.com/<your-username>/schrodinger-cn-fno.git
cd schrodinger-cn-fno
pip install numpy scipy jax flax optax einops pandas matplotlib
jupyter notebook notebook.ipynb
```

**Main dependencies:** `numpy`, `scipy`, `jax`, `flax`, `optax`, `einops`, `pandas`

---

## References

The Crank-Nicolson implementation is strongly inspired by the exposition of **Qijing Zheng** on numerical solutions to the TDSE:
> Zheng, Q. *Numerical Solutions to the Time-dependent Schrödinger Equation*. [staff.ustc.edu.cn](http://staff.ustc.edu.cn/~zqj/posts/Numerical_TDSE/)

The FNO architecture follows the seminal work:
> Li, Z. et al. *Fourier Neural Operator for Partial Differential Equations*. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

Additional references: Crank & Nicolson (1947), van Dijk & Toyama (2007), Tannor (2007), Conway (1990), Tao (2008).

---

## Author

**Juan Esteban Bejarano Aragón**  
Universidad El Bosque, Facultad de Matemáticas  
Bogotá, Colombia — 2026
