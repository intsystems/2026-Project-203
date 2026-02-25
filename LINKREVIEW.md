# LinkReview

- Here we collect all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them

> [!NOTE]
> This review table will be updated, so it is not a final version.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| SignSGD | signSGD: Compressed Optimisation for Non-Convex Problems | 2018 | Bernstein, J., et al. | [arXiv](https://arxiv.org/abs/1802.04434) | - | Introduces sign-based gradient descent to reduce communication costs in distributed training. |
| SignSGD | signSGD with Majority Vote is Communication Efficient And Fault Tolerant | 2019 | Bernstein, J., et al. | [arXiv](https://arxiv.org/abs/1810.05291) | - | Prove signSGD convergence in large and mini-batch settings (establishing for a specific parameter regime ADAM convergence as a byproduct) and demonstrate that majority vote aggregation ensures robustness against up to 50% adversarial workers.|
| Muon / LMO | Old Optimizer, New Norm: An Anthology | 2024 | Bernstein, J., Newhouse L. | [arXiv](https://arxiv.org/abs/2409.20325) | - | Connects classical optimization methods with modern norm-based constraints and preconditioning. |
| Muon / LMO | Muon: An Optimizer for Hidden Layers in Neural Networks | 2024 | Jordan, K., et al. | Blog | [GitHub](https://github.com/KellerJordan/Muon) | Proposes Muon optimizer specifically designed for hidden layers using matrix preconditioning. |
| Muon / LMO | Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization | 2025 | Kovalev, D. | [arXiv](https://arxiv.org/abs/2503.12645) | - | Provides theoretical analysis of gradient orthogonalization via Non-Euclidean Trust-Region Optimization. |
| Muon / LMO | The Ky Fan Norms and Beyond: Dual Norms and Combinations for Matrix Optimization | 2025 | Kravatskiy, A., et al. | [arXiv](https://arxiv.org/abs/2512.09678) | [GitHub](https://github.com/alexlegeartis/Neon/)| Utilizes non-spectral matrix norms to construct LMO-based algorithms |
| Muon / LMO | Deriving Muon | 2025 | Jeremy Bernstein | [Blog](https://jeremybernste.in/writing/deriving-muon) | - | Explains the theoretical derivation and intuition behind the Muon optimizer. |
| Muon / LMO | Muon is Scalable for LLM Training | 2025 | Liu, J., et al. | [arXiv](https://arxiv.org/abs/2502.16982 ) | - | Demonstrates Muon's scalability for LLM training, validating its performance against AdamW on large-scale models. |
| SignSGD | Stochastic Distributed Learning with Gradient Quantization and Double-Variance Reduction | 2023 | Horváth, S., et al. | [arXiv](https://arxiv.org/abs/1904.05115) | - | Provides theoretical analysis of gradient quantization and variance reduction in distributed learning for communication efficiency. |
| SignMuon | Leveraging Coordinate Momentum in SignSGD and Muon | 2025 | Petrov, E., et al. | [arXiv](https://arxiv.org/abs/2506.04430 ) | -| Zeroth-order modifications for SignSGD and Muon|
| Muon / LMO | A Note on the Convergence of Muon and Further | 2025 | Li, J. and Hong, M. | [arXiv](https://arxiv.org/abs/2502.02900 ) | - | Provides convergence analysis for Muon, addressing theoretical gaps in LMO-based optimizers. |
| Muon / LMO | The Polar Express: Optimal Matrix Sign Methods | 2025 | Amsel, N., et al. | [arXiv](https://arxiv.org/abs/2505.16932 ) | - | More effective replacement for Newton-Schulz iterations. |
| NormSGD | Momentum Improves Normalized SGD | 2020 | Cutkosky, A. and Mehta, H. | [arXiv](https://arxiv.org/abs/2002.03305) | - | Proves that momentum improves convergence for normalized SGD, offering theoretical insights relevant to sign-based methods. |
| Benchmarks | Practical Efficiency of Muon for Pretraining | 2025 | Shah, I., et al. | [arXiv](https://arxiv.org/abs/2505.02222 ) | - | Benchmarks Muon's practical efficiency in pretraining scenarios against standard baselines. |
| Benchmarks | Modded-NanoGPT: Speedrunning the NanoGPT Baseline | 2024a | Jordan, K., et al. | [GitHub](https://github.com/KellerJordan/modded-nanogpt ) | [GitHub](https://github.com/KellerJordan/modded-nanogpt ) | Establishes a fast NanoGPT baseline for comparing optimizer performance in LLM training. |
| Benchmarks | CIFAR-10 Airbench | 2023 | Jordan, K. | [GitHub](https://github.com/KellerJordan/cifar10-airbench) | [GitHub](https://github.com/KellerJordan/cifar10-airbench) | Standardized benchmark for comparing optimizer performance on CIFAR-10 with standardized architecture. |
| SignSGD | signSGD: Compressed Optimisation for Non-Convex Problems | 2018 | Bernstein, J., et al. | [arXiv](https://arxiv.org/abs/1802.04434 ) | - | - |
| Muon / LMO | Generalized Gradient Norm Clipping & Non-Euclidean (L_0,L_1)-Smoothness | 2025 | Pethick, T., et al. | [arXiv](https://arxiv.org/abs/2506.01913 ) | - | - |
| Muon / LMO | Dion: Distributed Orthonormalized Updates | 2025 | Ahn, K., et al. | [arXiv](https://arxiv.org/abs/2504.05295 ) | - | - |
| Muon / LMO | An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants | 2025 | Ahn, K., et al.  | [arXiv](https://arxiv.org/abs/2510.09827 ) | - | - |
| Compression | Error Feedback for Muon and Friends | 2025 | Gruntkowska, K., et al. | [arXiv](https://arxiv.org/abs/2510.00643 ) | - | - |
| Compression | Effective Quantization of Muon Optimizer States | 2025 | Gupta A., et al. | [arXiv](https://arxiv.org/abs/2509.23106 ) | - | - |
| Theory | Stochastic Spectral Descent for Restricted Boltzmann Machines | 2015 | Carlson D., et al. | [PMLR](https://proceedings.mlr.press/v38/carlson15.html ) | - | - |
| SingSGD | Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions | 2023 | Sun T., et al. | [PMLR](https://proceedings.mlr.press/v202/sun23l) | - | - |
| Theory | Improving dimensionality reduction with spectral gradient descent | 2005 | Hinton, G. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608005001310 ) | [PDF](https://www.cs.toronto.edu/~rfm/pubs/sg.pdf ) | - |


