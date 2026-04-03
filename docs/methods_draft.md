# 7 Methods

## 7.1 Dataset and Continuous State Formulation
The experiments utilize the `StrokeRehab` dataset, containing human rehabilitation motion trials. The state of the motion at any given discrete timestep is represented by a 103-dimensional feature vector $x \in \mathbb{R}^{103}$, encoding the multi-joint kinematic configuration. A full trajectory is represented as $X = \{x_1, x_2, \dots, x_T\}$.

To simulate real-world wearable sensor degradation, we formalize a corruption model $\mathcal{C}(X, c_{type}, \sigma)$, where $c_{type}$ defines the noise typology (e.g., Brownian drift, temporal dropout, or timing jitter) and $\sigma$ represents severity. The degraded input sequence is denoted as $X_{corr} = \mathcal{C}(X_{clean})$.

## 7.2 Physics-Constrained Flow Matching
Instead of traditional single-step denoising or standard diffusion formulations, we frame trajectory correction as continuous normalizing flows defined by an Ordinary Differential Equation (ODE). We aim to learn a time-conditioned velocity field $v_\theta(x,t)$ parameterized by a neural network $\theta$.

The target vector field maps the corrupted base distribution $p_0(x)$ toward the target clean distribution $p_1(x)$.
The loss function for conditional flow matching is defined over the continuous time variable $t \in [0, 1]$:

$$ \mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, p_t(x)} \left[ || v_\theta(x_t, t) - u_t(x_t) ||^2_2 \right] $$

where $u_t(x_t)$ is the optimal transport conditional vector field pointing from $x_{corr}$ to $x_{clean}$.

### 7.2.1 Kinematic Safety Regularizers
Pure numerical reconstruction does not guarantee biomechanical realism. To enforce physical plausibility, we extend the flow-matching objective with safety-oriented penalty constraints:

$$ \mathcal{L}_{total} = \mathcal{L}_{FM} + \lambda_{jerk}\mathcal{L}_{jerk} + \lambda_{limits}\mathcal{L}_{limits} $$

1. **Jerk Penalty ($\mathcal{L}_{jerk}$):** Minimizes the third temporal derivative of the generated trajectory to ensure smoothness:
   $$ \mathcal{L}_{jerk} = \sum_{t} || \dddot{X}_{pred}(t) ||^2_2 $$

2. **Joint-Limit Barrier ($\mathcal{L}_{limits}$):** An asymmetric ReLU-based penalty activated when the produced feature vector exceeds physically possible angular bounds.

## 7.3 Model Architectures
The system is bifurcated into two primary tracks:

### 7.3.1 ANN Flow Generator (Priority A)
The core backbone predicting $v_\theta$ is a 1D Temporal U-Net geometry. It ingests the noisy trajectory $X_{corr}$ concatenated with a binary validity mask. Time $t$ is injected into the intermediate layers using sinusoidal positional embeddings, allowing the capacity to resolve both broad sequence structure and high-frequency details. 

### 7.3.2 Hybrid ANN-SNN (Priority B)
For low-power, edge-deployed adaptation, we prototype a hybrid integration strategy:
- **ANN Student:** A parameter-efficient dense network (using depthwise/pointwise 1D convolutions) that executes the structurally continuous, low-frequency approximation of the trajectory.
- **SNN Residual:** An event-driven correction network utilizing Stateful Leaky Integrate-and-Fire (LIF) neurons. Operating on incremental inter-frame variances ($\Delta x$), the SNN explicitly generates sparse, high-frequency residual adjustments $\delta X_{snn}$ to rapidly suppress local aberrations.

The final trajectory estimation becomes $X_{hybrid} = X_{ANN} + \delta X_{snn}$, pairing computational efficiency with robust high-frequency correction.
