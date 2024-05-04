<p align="center" >
  <img width="50%" src="/docs/aigle_logo.png" />
</p>

# About AIGLETools
AIGLETools is a package written in Python, designed to minimize the effort required to build ab initio generalized Langevin equation (AIGLE) & ab initio Langevin equation (AILE) models for multi-dimensional time series data. 

### What is Generalized Langevin equation (GLE)
GLE is a non-Markovian equation of motion describing the time evolution of a system with generalized coordinates $\{\mathbf{x}, \mathbf{p}\}$:

$\dot{\mathbf{p}}(t) = -\nabla_{\mathbf{x}}G(\mathbf{x}(t))  +  \int_0^t \mathbf{K}(s)\mathbf{p}(t-s) ds  + \mathbf{R}(t) + \mathbf{\xi}(t);$

$\dot{\mathbf{x}}(t) = \mathbf{M}^{-1}\mathbf{p}(t)$.

For n-dimensional generalized position $\mathbf{x}$ and n-dimensional generalized momentum $\mathbf{p}$, $G(\mathbf{x})$ is the effective potential energy of the system. $\mathbf{\xi}(t)$ and $\mathbf{F}(t)$ are respectively the external driving force and the environmental noise. $\mathbf{K}(s)$ is a $n\times n$ memory kernel matrix, describing how the system at time $t$ responds to its historical state at time $t-s$. $M$ is a static $n\times n$ matrix, describing the inertia of the system. 

### What is Langevin equation (GLE)

The Langevin equation(LE) is the Markovian limit of GLE, given as

$\dot{\mathbf{p}}(t) = -\nabla_{\mathbf{x}}G(\mathbf{x}(t))  -  \eta \mathbf{p}(t)  + \mathbf{w}(t) + \mathbf{\xi}(t);$

$\dot{\mathbf{x}}(t) = \mathbf{M}^{-1}\mathbf{p}(t)$.

Here, $\eta$ is a static $n\times n$ matrix, mimicking a friction acting on the system. $\mathbf{w}(t)$ is a white noise. 


## What is AIGLE and AILE?
AIGLE is the generalized Langevin equation extracted from the history of $\mathbf{x}(t)$ with few or no _ad hoc_ assumptions. AILE is taken as the Markovian limit of AIGLE. Both AIGLE and AILE suppose to recover dynamical properties of $\mathbf{x}(t)$, while AIGLE is expected to be more faithful to the time series data than AILE.

# Highlighted features

- Deal with high-dimensional and heterogeneous time-series data  

- Integrate data processing, model training and simulation 
  - User-friendly training of GLE model; Expertise in GLE not required.
  - Built-in multi-dimensional GLE/LE simulator
  - OPENMM plugin (Python interface)

- Exact enforcement of second fluctuation-dissipation theorem for long-term simulation

# Credits
In the future, please cite our incoming preprint for general purpose.

# Installation
pip install .

# Use AIGLETools
The training and simulation of AIGLE/AILE model follows the steps below

<p align="center" >
  <img width="100%" src="/docs/workflow.png" />
</p>


# Examples
Check /examples for Jupyter notebook demonstration of case studies!

<!-- ## Units -->

