# LinkReview

- Here we collect all the works that may be useful for writing our paper
- We divide these works by topic in order to structure them

> [!NOTE]
> This review table will be updated, so it is not a final version.

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| SignSGD | signSGD: Compressed Optimisation for Non-Convex Problems | 2018 | Bernstein, J., et al. | [arXiv](https://arxiv.org/abs/1802.04434) | GitHub | Introduces sign-based gradient descent to reduce communication costs in distributed training. |
| SignSGD | signSGD with Majority Vote is Communication Efficient And Fault Tolerant | 2019 | Bernstein, J., et al. | [arXiv](https://arxiv.org/abs/1810.05291) | GitHub | Prove signSGD convergence in large and mini-batch settings (establishing ADAM convergence as a byproduct) and demonstrate that majority vote aggregation ensures robustness against up to 50% adversarial workers.|
| Muon / LMO | Old Optimizer, New Norm: An Anthology | 2024 | Bernstein, J., Newhouse L. | [arXiv](https://arxiv.org/abs/2409.20325) | GitHub | Connects classical optimization methods with modern norm-based constraints and preconditioning. |
| Muon / LMO | Muon: An Optimizer for Hidden Layers in Neural Networks | 2024 | Jordan, K., et al. | Blog | [GitHub](https://github.com/KellerJordan/Muon) | Proposes Muon optimizer specifically designed for hidden layers using matrix preconditioning. |
| Muon / LMO | Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization | 2025 | Kovalev, D. | [arXiv](https://arxiv.org/abs/2503.12645) | GitHub | Provides theoretical analysis of gradient orthogonalization via Non-Euclidean Trust-Region Optimization. |
| Muon / LMO | The Ky Fan Norms and Beyond: Dual Norms and Combinations for Matrix Optimization | 2025 | Kravatskiy, A., et al. | [arXiv](https://arxiv.org/abs/2512.09678) | GitHub | Explores Ky Fan norms and dual norm combinations for advanced matrix optimization tasks. |
| Muon / LMO | Deriving Muon | 2025 | Jeremy Bernstein | [Blog](https://jeremybernste.in/writing/deriving-muon) | GitHub | Explains the theoretical derivation and intuition behind the Muon optimizer. |
| Muon / LMO | Muon is Scalable for LLM Training | 2025 | Liu, J., et al. | [arXiv](https://arxiv.org/abs/2502.16982 ) | GitHub | Demonstrates Muon's scalability for LLM training, validating its performance against AdamW on large-scale models. |
| SignSGD | Stochastic Distributed Learning with Gradient Quantization and Double-Variance Reduction | 2023 | Horváth, S., et al. | [arXiv](https://arxiv.org/abs/1904.05115) | GitHub | Provides theoretical analysis of gradient quantization and variance reduction in distributed learning for communication efficiency. |
| SignSGD | Momentum Improves Normalized SGD | 2020 | Cutkosky, A. and Mehta, H. | [arXiv](https://arxiv.org/abs/2002.03305) | GitHub | Proves that momentum improves convergence for normalized SGD, offering theoretical insights relevant to sign-based methods. |
| SignMuon | Leveraging Coordinate Momentum in SignSGD and Muon | 2025 | Petrov, E., et al. | [arXiv](https://arxiv.org/abs/2506.04430 ) | GitHub | Explores coordinate momentum techniques applicable to both SignSGD and Muon for memory optimization and efficiency. |
| Muon / LMO | A Note on the Convergence of Muon and Further | 2025 | Li, J. and Hong, M. | [arXiv](https://arxiv.org/abs/2502.02900 ) | GitHub | Provides convergence analysis for Muon, addressing theoretical gaps in LMO-based optimizers. |
| Benchmarks | Practical Efficiency of Muon for Pretraining | 2025 | Shah, I., et al. | [arXiv](https://arxiv.org/abs/2505.02222 ) | GitHub | Benchmarks Muon's practical efficiency in pretraining scenarios against standard baselines. |
| Muon / LMO | The Polar Express: Optimal Matrix Sign Methods | 2025 | Amsel, N., et al. | [arXiv](https://arxiv.org/abs/2505.16932 ) | GitHub | Analyzes matrix sign methods and their theoretical connection to the Muon algorithm. |
| Benchmarks | Modded-NanoGPT: Speedrunning the NanoGPT Baseline | 2024a | Jordan, K., et al. | [GitHub](https://github.com/KellerJordan/modded-nanogpt ) | [GitHub](https://github.com/KellerJordan/modded-nanogpt ) | Establishes a fast NanoGPT baseline for comparing optimizer performance in LLM training. |


