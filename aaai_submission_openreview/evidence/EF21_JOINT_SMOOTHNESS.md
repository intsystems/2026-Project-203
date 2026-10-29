# Candidate EF21 derivation for coupled layers: internal review only

Status after revision 2 audit: **separate candidate analysis, not an established revised-manuscript result and not relied on in the rebuttals**. The current EF21 manuscript proofs explicitly retain block separability, and the author addendum marks cross-layer coupling open. The calculations below were prepared separately by the agent; author mathematical review and incorporation are outstanding. They must not be cited as evidence that the paper has already removed the assumption. The manuscript’s conditional plain SignMuon theorem does use joint smoothness. The original note is preserved in the revision 2 backup.

## 1. A direct reduction removes the separability requirement

Let the full model be the tuple \(X=(X_\ell)_\ell\), with positive fixed layer multipliers \(\lambda_\ell\), and put

\[
 \|X\|_\lambda=\max_\ell\frac{\|X_\ell\|_{\rm op}}{\lambda_\ell},\quad
 \mathcal N_\lambda(G)=\sum_\ell\lambda_\ell\|G_\ell\|_*,\quad
 B^2=\sum_\ell\lambda_\ell^2\min(m_\ell,n_\ell).
\]

The second quantity is exactly the dual of the first. The LMO over its unit ball returns the tuple \(-\lambda_\ell\operatorname{polar}(G_\ell)\), with the zero selection at a zero input. This is the same collection of layer updates as before, with one common scalar learning rate. It is a **single block in the existing arbitrary-norm EF21 framework**. It does not assume that the objective is a sum of functions of individual layers.

For the joint Frobenius norm, \(\|X\|_F^2=\sum_\ell\|X_\ell\|_F^2\),

\[
 (\min_\ell\lambda_\ell)\|X\|_\lambda\le\|X\|_F\le B\|X\|_\lambda,
 \qquad\mathcal N_\lambda(G)\le B\|G\|_F.
\]

If each local gradient is jointly Lipschitz in Frobenius norm with constant \(L_j\), then

\[
 \mathcal N_\lambda(\nabla f_j(X)-\nabla f_j(Y))
 \le B^2 L_j\|X-Y\|_\lambda.
\]

Thus the framework's ordinary smoothness assumption holds for the full tuple, with constant \(B^2L_j\). A Frobenius noise variance bound \(\sigma^2\) implies the required dual-norm variance bound \(B^2\sigma^2\). Compress each layer separately by \(C_\ell(Y)=\operatorname{mean}|Y|\operatorname{sign}(Y)\). Summing the submitted Lemma 5 over layers proves

\[
 \|C(Y)-Y\|_F^2\le(1-\alpha)\|Y\|_F^2,
 \qquad \alpha=\min_\ell(m_\ell n_\ell)^{-1}>0.
\]

Consequently the original EF21 framework's one-block theorems apply to coupled objectives. The squared-dual-norm rate still requires its gradient-scaled radius, now a **common radius scaled by \(\mathcal N_\lambda(h_t)\)**. It does not automatically apply to the experiments' constant or cosine radius. Grouping the layers may worsen dimension constants. No claim of improved constants or a new general EF21 framework is justified by this reduction.

Primary framework: [Gruntkowska et al., Error Feedback for Muon and Friends](https://arxiv.org/html/2510.00643v1), Algorithm 3, Theorem 19, and the Euclidean downlink extension in Remark 23. The submitted paper already uses these results in Appendix A.10. The new observation is to apply them to one norm on the **whole tuple**, instead of their restrictive individual-block smoothness assumption.

## 2. An explicit first-moment bound for deterministic schedules

The following also extends the article's `ef21_fixed_radius.tex` proof directly, using ordinary joint Frobenius smoothness. Let \(f=N^{-1}\sum_j f_j\ge f_*\), with joint Frobenius smoothness constant \(L\), and let each \(f_j\) have constant at most \(\widehat L\). At every round, worker noises are conditionally centered, have conditional Frobenius variance at most \(\sigma^2\), and are independent across workers conditional on the common past. No independence between layers is required. No bound on gradient dissimilarity between clients is required.

Initialize \(X_0=W_0\), \(M_0^j=g_0^j=0\). With fixed \(\beta\in(0,1]\), \(\mu=1-\beta\), and \(\theta=0\) (EMA) or \(\theta=\beta\) (Nesterov), run

\[
\begin{aligned}
 A_t^j&=\nabla f_j(W_t)+\xi_t^j,\\
 M_{t+1}^j&=\mu M_t^j+\beta A_t^j,\qquad V_t^j=(1-\theta)M_{t+1}^j+\theta A_t^j,\\
 g_{t+1}^j&=g_t^j+C^\uparrow(V_t^j-g_t^j),\quad h_t=N^{-1}\sum_jg_{t+1}^j,\\
 X_{t+1,\ell}&=X_{t,\ell}-\eta_t\lambda_\ell\operatorname{polar}(h_{t,\ell}),\\
 W_{t+1}&=W_t+C^\downarrow(X_{t+1}-W_t).
\end{aligned}
\]

Let the two compressors satisfy the pathwise Frobenius error bounds \(\|R-C^\uparrow(R)\|_F\le q_U\|R\|_F\), \(\|R-C^\downarrow(R)\|_F\le q_P\|R\|_F\), where \(q_U,q_P<1\). Scaled sign satisfies these bounds, as does the identity with \(q_P=0\). Put \(\kappa_U=q_U/(1-q_U)\), \(\kappa_P=q_P/(1-q_P)\), \(\bar\eta=\max_{t<T}\eta_t\), \(S=\sum_{t<T}\eta_t>0\), \(Q=\sum_{t<T}\eta_t^2\), and

\[
\begin{aligned}
 k&=\beta+\mu\theta,\quad v=\beta/(2-\beta),\quad v_\theta=k^2+(1-\theta)^2\mu^2v,\\
 I_\beta&=(1-\theta)\mu\|\nabla f(X_0)\|_F/\beta
                 +\kappa_U N^{-1}\sum_j\|\nabla f_j(X_0)\|_F,\\
 D_\beta&=\widehat L B[\kappa_P+(1-\theta)\mu/\beta+\kappa_U(1+2k\kappa_P)],\\
 R_\beta&=\sigma[\sqrt{v_\theta/N}+\kappa_U((1-\theta)\beta(1+\sqrt v)+2\theta)].
\end{aligned}
\]

Then, with \(\Delta=f(X_0)-f_*\),

\[
 \frac{\sum_{t<T}\eta_t\,\mathbb E\mathcal N_\lambda(\nabla f(X_t))}{S}
 \le\frac{\Delta+2B\bar\eta I_\beta}{S}
       +2B(\bar\eta D_\beta+R_\beta)+\frac{LB^2Q}{2S}.
\tag{J}
\]

### Proof

**Downlink tracking.** The exact update has Frobenius length at most \(B\eta_t\). Contractivity gives \(\|X_t-W_t\|_F\le P:=\kappa_P B\bar\eta\) by induction, starting from a zero gap.

**Momentum tracking.** Split each EMA into its gradient and noise sums. Conditional centering makes noise increments orthogonal in expectation. The local EMA noise variance is at most \(\sigma^2v\). After averaging workers it is at most \(\sigma^2v/N\). The noise component of \(V_t^j\) is \((1-\theta)\mu Z_t^j+k\xi_t^j\), whose averaged variance is at most \(\sigma^2v_\theta/N\). The initial gradient contribution in the average is \(-(1-\theta)\mu^{t+1}\nabla f(X_0)\). Joint smoothness bounds the remaining target bias by \(\widehat LP+(1-\theta)\mu\widehat LB\bar\eta/\beta\). Hence

\[
 \mathbb E\|N^{-1}\sum_jV_t^j-\nabla f(X_t)\|_F
 \le(1-\theta)\mu^{t+1}\|\nabla f(X_0)\|_F
      +\widehat LP+(1-\theta)\mu\widehat LB\bar\eta/\beta
      +\sigma\sqrt{v_\theta/N}.
\]

All local gradient increments are bounded by displacement of the **entire model**, which is the change needed to avoid separability.

**Uplink tracking.** For \(t\ge1\),

\[
 V_t^j-V_{t-1}^j=(1-\theta)\beta(A_t^j-M_t^j)+\theta(A_t^j-A_{t-1}^j).
\]

Its expected norm is at most \((1-\theta)\beta\mu^t\|\nabla f_j(X_0)\|_F+J\), where

\[
 J=\widehat LB\bar\eta+2k\widehat LP
      +\sigma[(1-\theta)\beta(1+\sqrt v)+2\theta].
\]

Unroll the compressor recursion with multiplier \(q_U\). The total deterministic initial-target impulse is bounded by \([k+(1-\theta)\beta\sum_{t\ge1}\mu^t]\|\nabla f_j(X_0)\|_F=\|\nabla f_j(X_0)\|_F\). Initial noise is covered by \(J\), since \(k\le(1-\theta)\beta(1+\sqrt v)+2\theta\). Multiply by \(\eta_t\le\bar\eta\), sum the geometric initial terms, and bound the steady terms pointwise. Combining with the target bound gives

\[
 \sum_{t<T}\eta_t\mathbb E\|h_t-\nabla f(X_t)\|_F
 \le\bar\eta I_\beta+S(\bar\eta D_\beta+R_\beta).
\]

**Descent.** The exact LMO pairing and \(\mathcal N_\lambda(E)\le B\|E\|_F\) imply

\[
 \sum_\ell\lambda_\ell\langle\nabla_\ell f(X_t),\operatorname{polar}(h_{t,\ell})\rangle
 \ge\mathcal N_\lambda(\nabla f(X_t))-2B\|h_t-\nabla f(X_t)\|_F.
\]

Apply ordinary joint Frobenius smoothness, whose remainder is at most \(LB^2\eta_t^2/2\), take expectations, and telescope the objective gap. Substitute the tracking bound and divide by \(S\). This proves (J).

## 3. Rates and the exact experimental limitation

For a constant schedule \(S=T\bar\eta\). For cosine \(\eta_t=\bar\eta[1+\cos(\pi t/T)]/2\), \(S=\bar\eta(T+1)/2\). In both cases \(Q/S\le\bar\eta\). As \(\beta\downarrow0\), \(I_\beta,D_\beta=O(\beta^{-1})\), \(R_\beta=O(\sqrt\beta)\). Taking \(\beta=T^{-1/2}\) and \(\bar\eta=\eta_0T^{-3/4}\), with fixed positive \(\eta_0\), therefore gives a learning-rate-weighted **first moment** of order \(O(T^{-1/4})\) for both EMA/Nesterov and either downlink. A fixed \(\beta\) and fixed peak learning rate instead leave an explicit residual bound.

If validated and adopted, this proposed extension would remove layer separability for the specified ideal EF21 algorithms. It is not the theorem currently proved in the revised manuscript. It establishes neither a squared-gradient rate for those schedules nor an end-to-end guarantee for a network run using a finite polynomial polar solver, changing momentum, or auxiliary Adam updates. The nanoGPT experiment has all three of those implementation qualifications. All guarantees here concern the server iterate \(X_t\); successful validation at \(W_t\) does not establish success at \(X_t\).
