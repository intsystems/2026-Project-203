# Confidential comment to AC.

Dear Area Chair,

Thank you for your time and consideration. We are writing to express several concerns regarding the reviewers’ feedback, specifically related to reviewers xNrk and mLLq. We believe that their final justifications do not fully reflect the contribution and context of our work

## Reviever xNrk

From our perspective, the discussion with Reviewer xNrk evolved as follows. After our rebuttal, the reviewer explicitly stated:

1) “a) - fully resolved concerns”; 
2) “I have no further comments” 

We therefore understood that the issues had been addressed. However, the final justification states that “my primary concern remains the scope and overall contribution of the work,” without additional clarification. This shift is difficult for us to interpret and makes it unclear how the remaining concerns could have been addressed.

Regarding the substance of this concern, we would like to emphasize that our work addresses a well-motivated and previously unresolved question: the high-probability convergence of Clip-SGD under the joint assumptions of heavy-tailed noise and $(L_0, L_1)$-smoothnesss. 

We also respectfully disagree that having a single main theorem is a weakness. In theoretical work, a single central result can constitute a strong contribution when it addresses an open question. Our theorem provides the first guarantees in this setting and improves upon prior results (e.g., avoiding exponential dependence on $L_1 R_0$). Moreover, during the discussion phase, we expanded the presentation to include additional perspectives, such as non-convex considerations and probability amplification techniques, and other reviewers acknowledged these improvements (e.g., **reviewer vNTE increased their score**, and **reviewer Uqxp provided a supportive evaluation**).

Finally, we note that the importance and current relevance of this topic are supported by numerous references in the paper, reflecting active interest in areas such as heavy-tailed noise, generalized smoothness, and gradient clipping.

<!--- 
We believe that the contribution of our work is very important - our study is related to topics such as generalized smoothness, heavy-tailed stochasticity, high-probability convergence, and gradient clipping. All of these topics are currently important and actively studied; moreover, some fundamental questions are still open for the Clipped-SGD. And, indeed, our paper answers on some of them. 
--->

<!--- 
In addition, we can say that reviewer xNrk *is not correct* in claiming that the paper presents only a single theorem: during the discussion with the reviewers (reviewer vNTE increased the final score from 4 to 5, and reviewer Uqxp gave a score of 4), we expanded our contribution by adding

- Non-convex analysis;

- Discussion related to probablility amplification.
--->

## Reviewer mLLq

We are glad that the Reviewer was able to engage in the discussion with us. Our only concern is the reviewer’s final assessment of our contribution. 

***We respectfully disagree with the conclusion that our contribution is limited to proving only a single regime, as stated by the reviewer***. We obtained the *only existing upper bounds* for the setting under consideration, and the fact that they are somewhat nonstandard does not mean that there is no result at all. As also noted by the reviewer, the paper clearly states its assumptions and limitations. Importantly, prior to our work, there were neither matching upper bounds nor lower bounds without exponential dependence in this setting (e.g., as seen in Sadiev et al., 2023). From this perspective, establishing any meaningful and nontrivial guarantees already represents significant progress.

---

We would like to emphasize that we will incorporate all clarifications and suggestions that emerged during the discussion into the final version of the paper. We are also **sincerely grateful to Reviewers vNTE and Uqxp for their constructive feedback and engagement**.

We fully **respect the decision-making process and trust your judgment**. Our intention in writing this note is simply to provide additional context on the discussion dynamics and to ensure that the final decision is made with a complete and accurate picture of the contributions and reviewer interactions.

Thank you again for your time and consideration.

Best regards,

Authors

# Second lap: responces to reviewers

# Reviewer vNTE

Dear Reviewer,

Thank you for carefully reading our response. We are happy to further clarify the remaining concerns and will include the following additions in the final version.

>***Non-convex analysis.***

We outline the non-convex analysis pipeline that will be incorporated into the paper.

1) **Descent lemma.** Following Nguyen et al. (2023), under the assumptions $||\nabla f(x_k)|| \leq \frac{\lambda}{2}$ and $\gamma \lesssim \frac{1}{L_0 + L_1\lambda}$, we obtain
\begin{align*}
    f(x_{k+1}) - f^\star &\lesssim f(x_k) - f^\star - \frac{\gamma}{2}||\nabla f(x_k)||^2 + \gamma||\theta_k^b||^2 \\&- (\gamma -(L_0 + L_1||\nabla f(x_k)||)\gamma^2)\langle\nabla f(x_k), \theta_k^u \rangle \\&+ (L_0 + L_1||\nabla f(x_k)||)\gamma^2 ||\theta_k^u||^2.
\end{align*}
This holds only when $||\nabla f(x_k)|| \leq \frac{\lambda}{2}$, ensured with high probability later.
2) **Events $E_k$** are defined as follows:
\begin{align*}
    \sum\limits_{l=0}^{t-1}\gamma||\theta_l^b||^2 &- (\gamma -(L_0 + L_1||\nabla f(x_l)||)\gamma^2)\langle\nabla f(x_l), \theta_l^u \rangle \\&+ (L_0 + L_1||\nabla f(x_l)||)\gamma^2 ||\theta_l^u||^2 \leq \Delta_0 = f(x_0) - f^\star;\\
    \Delta_t &\leq 2\Delta_0;\\
    ||\nabla f(x_t)|| &\leq \frac{\lambda}{2}
\end{align*}
for all $t=0, \ldots, k$
3) **Parameters.** 
$$\gamma \simeq \min\left\{\frac{\Delta_0}{(L_1\Delta_0 + \sqrt{L_0\Delta_0})^2K^{\frac{\alpha-1}{3\alpha-2}}}, \frac{p(L_1, L_0, \Delta_0, \sigma)}{K^{\frac{\alpha}{3\alpha-2}}}\right\},$$
$$\lambda \simeq \frac{\Delta_0}{(L_1\Delta_0 + \sqrt{L_0\Delta_0})\gamma K^{\frac{\alpha-1}{3\alpha-2}}}.$$
(Log factors and constants are omitted; the second term extends Nguyen et al.)
4) **Key steps in the proof.**
- By induction, $E_{T-1}$ ensures $||\nabla f(x_t)|| \le \frac{\lambda}{2}$, so the descent lemma applies.
- Clipping lemma (Lemma 1) is valid.
- Using Proposition 1,
$$||\nabla f(x_T)||^2 \leq 4(L_0 + L_1||\nabla f(x_T)||)(f(x_T) - f^\star).$$
which yields $||\nabla f(x_T)|| \lesssim L_1\Delta_0 + \sqrt{L_0\Delta_0}$ in the event $E_T$ after solving the quadratic inequality. This ensures $|\nabla f(x_T)| \le \frac{\lambda}{2}$ and closes induction.
5) **The final bound** is:
$$\frac{1}{K}\sum\limits_{k=0}^{K-1}||\nabla f(x_k)||^2 = \tilde{\mathcal{O}}\left(\frac{L_0\Delta_0}{K^{\frac{2\alpha - 1}{3\alpha - 2}}} + \frac{(L_1\Delta_0)^2}{K^{\frac{2\alpha - 1}{3\alpha - 2}}} + \frac{\Delta_0}{p(L_1, L_0, \Delta_0, \sigma)K^{\frac{2\alpha - 2}{3\alpha - 2}}}\right)$$
with probablitity at least $1 - \delta$, matching Nguyen et al. (2023) in $K$.

Importantly, the non-convex case is simpler: dependence on $\Delta_0$ is unavoidable even in deterministic smooth settings (Carmon et al., 2020), allowing $\lambda$ to depend on $\Delta_0$. This is not viable in the convex case due to exponential dependence on $L_1R_0$.

Carmon et al. "Lower bounds for finding stationary points I." Mathematical Programming 2020.

<!--- 
It it worth emphasizing that in the non-convex case, ***we do not have a problem with the guarantee*** $||\nabla f(x_k)|| \leq \frac{\lambda}{2}$; it significantly simplifies the proof due to the hidden exponential dependency of $\Delta_0 \sim \exp(R_0)$. In contrast, for the convex case, we have
$$||\nabla f(x_k)|| \leq L_0R_k\exp(L_1R_k).$$ 
-->

>***Depencence on $\delta$.***

We will add a point about the probability amplification trick to the final version of the paper as a tool that makes it possible to obtain a “standard” high-probability convergence bound. However, we do have some concerns about the possibility of deriving results you mentioned. Indeed, let us compare our results for the convex scenario with the existing papers; to be more precise, additional terms in the final bound.

- Deterministic (Gorbunov et al. (2024)): ${\mathcal{O}}\left((L_1R_0)^2\right);$

- Light-tailed noise (Gaash et al. (2025)): ${\mathcal{O}}\left((L_1R_0)^2\log\left(\frac{1}{\delta}\right)\right);$

- Heavy-tailed noise (our paper): ${\mathcal{O}}\left(\frac{(L_1R_0)^2}{\delta}\right)$
(here we decrease the power $2 + \alpha \rightarrow 2$ since we believe that it is the artifact of the proof and it can be handled). 

Moreoever, the last two results were presented as ***ergodic*** convergence over a *good* set (see the inequality above (30)), and this immediately leads to two points: 
1) It is *unfair* to compare our results with the probability amplification trick -- it changes the final convergence criterion ($\min$ over all runs);
2) It is possible that our result is *non-improvable* as a natural transition between the level of stochastisity. We emphasize that there exist no lower bounds neither upper bounds (in-expectation/high-probability) for the setup we consider.

Therefore, there is still a big open question -- is it possible to derive a better converegence guarantees for a convex setting? For now, **our paper is the only one dealing with this setting and ensuring the convergence without exponential dependency on $R_0$**.

**We hope that the additional non-convex results to be included in the final version, along with the clarifications regarding $\delta$, will help address your concerns and lead to a reconsideration of the overall evaluation of our work. We sincerely thank you for your valuable suggestions, which have helped improve the paper.**


(немного перебрал, 500 символов лишние)

# Reviewer Uqxp

Thank you for your comments. We are happy to further clarify the remaining concerns.

>***In-expectation results.*** 

We partially agree with your observation. The $\frac{1}{\delta}$ factor indeed stems from our high-probability analysis; however, it is currently unclear what the corresponding bound in expectation would be in this setting. While we understand the concern that our result may not outperform what could be obtained via probability amplification applied to an expectation bound, we would like to emphasize that our comparison was intended to be conceptual rather than literal. **To the best of our knowledge, no expectation-based guarantees (without exponential dependency on $L_1 R_0$) exist for this problem**. We will clarify this point in the final version and explicitly note that, in general, more refined approaches than a direct application of Markov’s inequality may be possible.

We would also like to highlight that, in some cases, deriving high-probability guarantees can be more tractable than obtaining bounds in expectation. In our analysis, the inductive argument allows us to effectively operate within a compact region by showing that the iterates remain in such a region on the “good” event. This property is crucial for controlling the behavior of the algorithm. In contrast, expectation-based analyses do not provide an analogous mechanism, making them substantially more challenging and likely requiring different techniques altogether.

>***Iterate selection.*** 

We fully agree that this is a limitation of our work. We hope that future research will help address this issue, either through a more sophisticated scheme for selecting the point or through a full ergodic convergence result. However, we would like to emphasize that **our paper is the only one dealing with the setting (convex, $(L_0,L_1)$-smooth optimization with heavy-tailed noise) and ensuring the convergence without exponential dependency on $R_0$**.

>***Dangling $\frac{1}{\delta}$***

We acknowledge that this term is non-standard. However, we conjecture that it *may be unimprovable*, although we cannot claim this definitively. If we consider the convex scenarion and compare the deterministic bound (Gorbunov et al.(2024)), the stochastic case with light-tailed noise (Gaash et al. (2025)), and the stochastic case with heavy-tailed noise (our result), we obtain the following comparison:

- Deterministic (Gorbunov et al. (2024)): ${\mathcal{O}}\left((L_1R_0)^2\right);$

- Light-tailed noise (Gaash et al. (2025)): ${\mathcal{O}}\left((L_1R_0)^2\log\left(\frac{1}{\delta}\right)\right);$

- Heavy-tailed noise (our paper): ${\mathcal{O}}\left(\frac{(L_1R_0)^2}{\delta}\right)$
(here we decrease the power $2 + \alpha \rightarrow 2$ since we believe that it is the artifact of the proof). Term $\frac{1}{\delta}$ may well be not an artifact of the proof, but rather a *natural factor* arising from the level of stochasticity. Since there are neither lower bounds nor upper bounds in expectation, it is impossible to determine whether this factor can be eliminated in terms of ergodic convergence over a "good" set (which is precisely the criterion used in the last two papers; see also the inequality above (30) in our paper). In addition, even if probability amplification allows one to get rid of factor $\frac{1}{\delta}$, one should not forget that the convergence criterion changes as well (a minimum over all runs appears).

We sincerely thank you for engaging in this discussion and for your insightful comments, which have helped us improve the paper. We would also like to note that we have developed a non-convex analysis (which, as expected, is technically simpler than the convex case; see our response to Reviewer vNTE) and will include it in the final version. We hope that these clarifications are helpful and that our work can serve as a foundation for further research in this important direction.


# Reviewer mLLq (second short responce, PLEASE WATCH THIS)

Dear Reviewer,

Thank you for your careful reading of our response. We would like to clarify several points that may not have been sufficiently clear.

>***Regarding $\frac{1}{\delta}$ and probability amplification***

The analysis in our paper is developed **without** probability amplification. We mentioned it only to illustrate that the $\frac{1}{\delta}$ factor can be removed *if one changes the setting* (multiple runs). This does **not** resolve the issue for a single run, which is the focus of our work.

We emphasize two key aspects.

***Comparison with existing high-probability results.*** We agree that standard high-probability bounds usually contain at most $\log(1/\delta)$. The key question, however, is what our result should be compared with.

For this setup:

1. **No convergence bound in expectation is known.** 
2. **The only existing high-probability result is Sadiev et al.**

Our bound is

$$K=\tilde{\mathcal O}\left(\frac{L_0R_0^2}{\varepsilon}
+\left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}}
+\frac{(L_1R_0)^{2+\alpha}}{\delta}\right),$$
while Sadiev et al. (2023) obtain
$$K=\tilde{\mathcal O}\left(\frac{L_0(1+L_1R_0\exp({L_1R_0}))R_0^2}{\varepsilon}
+\left(\frac{R_0\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}}\right).$$


<!---
To highlight the role of the “problematic” term, assume $L_1R_0\gg1$. Then the comparison is essentially
--->

Focusing on leading terms:
$$\left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}}
+\frac{(L_1R_0)^{2+\alpha}}{\delta}
\quad \text{vs.} \quad
\frac{L_0(1+L_1R_0\exp({L_1R_0}))R_0^2}{\varepsilon}
+\left(\frac{R_0\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}}.$$
<!---
$$\left((L_1R_0)^{\frac{\alpha}{\alpha-1}}-1\right)
\left(\frac{R_0\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}}
+\frac{(L_1R_0)^{2+\alpha}}{\delta}
\quad \text{vs.} \quad
\frac{L_0(L_1R_0\exp({L_1R_0}))R_0^2}{\varepsilon}.$$
--->

<!---
Because many parameters are involved, direct comparison is not very informative; a numerical example is more revealing for us.  For instance, let us take $\alpha = 3/2$, $\sigma = 1$, $L_0 = 1$, $L_1 = 10$, $R_0 = 10$, and two choices of $\varepsilon$: $\varepsilon = 10^{-10}$ (very good optimization error) and $\varepsilon = 10^{-1}$ (very poor optimization error). If we begin comparing the bounds in the order of the terms, we obtain
--->

A numerical illustration ($\alpha=3/2$, $L_0=1$, $L_1=10$, $R_0=10$, $\sigma=1$) shows:
$$10^{39} + \frac{10^7}{\delta} \text{ vs } 2.7 \cdot 10^{58} + 10^{33}\quad \text{when } \varepsilon = 10^{-10}$$
and
$$10^{12} + \frac{10^7}{\delta} \text{ vs } 2.7 \cdot 10^{49} + 10^6\quad \text{when } \varepsilon = 10^{-1}$$
As a direct result, our bound will be *worse* than the bound from Sadiev et al. (2023) in the case
$$\delta \lesssim 0.4 \cdot 10^{-51}\quad \text{when } \varepsilon = 10^{-10}$$
and
$$\delta \lesssim 0.4 \cdot 10^{-42}\quad \text{when } \varepsilon = 10^{-1}.$$
In both cases, the provided bounds for $\delta$ are **negligible**, i.e., **for all practically meaningful choices of $\delta$ our bound is much better than the previous state-of-the-art one in this case**. This illustrates our main contribution: **avoiding the exponential dependence** $\exp({L_1R_0})$, which is the central issue addressed in our work. To the best of our knowledge, our paper is the *first* to obtain a high-probability result for the convex generalized-smooth setting with heavy-tailed noise **without exponential dependence on $R_0$**.

***Nature of the stochasticity.*** If one considers all results for the convex case under generalized smoothness without using probability amplification, then, besides our work, there are two papers that can be compared in terms of the additional term:

- Deterministic (Gorbunov et al. (2024)): ${\mathcal{O}}\left((L_1R_0)^2\right);$

- Light-tailed noise (Gaash et al. (2025)): ${\mathcal{O}}\left((L_1R_0)^2\log\left(\frac{1}{\delta}\right)\right);$

- Heavy-tailed noise (our paper): ${\mathcal{O}}\left(\frac{(L_1R_0)^2}{\delta}\right)$
(here we decrease the power $2 + \alpha \rightarrow 2$ since we believe that it is the artifact of the proof). At present, ***neither the lower bounds nor even the in-expectation upper bounds*** are known. Therefore, term $\frac{1}{\delta}$ may in fact be *unimprovable*: it may be caused by the very nature of the stochasticity rather than by an imprecise analysis. Additionally, comparing with probability amplification method directly is **simply incorrect**: applying probability amplification changes the convergence criterion (a minimum over all runs appears), whereas in the above-mentioned works and in our paper the convergence criterion for a single run is *last-iterate* for the first one and *ergodic* over a "good" set for the last two results (for our paper, see the result from appendix, right above inequality (30)).

Therefore, the question remains open: is it possible to derive a "classical" (i.e., without $\frac{1}{\delta}$ factors) high-probability bound for convex $(L_0,L_1)$-smooth problems with heavy-tailed noise **without exponential dependence on $R_0$**? We believe our paper, as the first to address this problem, provides a useful starting point for future work.

>***Classical $(L_0,L_1)$-smoothness***

We hope this concern is now fully addressed. The assumption used in our paper is relevant and standard in recent literature.

We also note that we have prepared a nonconvex analysis, which, as expected, is easier than the convex case (see our responce to Reviewer vNTE).

Thank you again for your thoughtful comments. We hope that these clarifications address your concerns and will be taken into account in your final evaluation.


# Reviewer mLLq

Dear Reviewer,

Thanks for the carefully reading of our responce. To address your concerns, we need to clarify several points, as we are concerned that not all of our arguments were understood correctly.

>***Regarding $\frac{1}{\delta}$ and amplification technique***

Our goal is *not to apply the probability amplification technique*. On the contrary, our analysis in the paper is developed ***without this mechanism***. In our response, we showed that the amplification technique makes it possible to eliminate the problematic factor $\frac{1}{\delta}$ that concerned you. This ***does not mean*** that we have resolved this issue for a single run of the algorithm.

Nevertheless, we would like to elaborate on the contributions of our work. We believe it is necessary to discuss two fundamental points, after which we hope our contribution will become clearer.

***Comparison with the existing high-probability results.*** We naturally agree that the *standard* acceptable high-probability convergence bounds should contain at most a factor of $\log(\frac{1}{\delta})$. The obvious question, however, is: what bounds should our result be compared against? We would like to emphasize that:

1) **There does not exist a convergence bound in expectation for this setup.**
2) **The only high-probability convergence bound for this setup is Sadiev et al.** For clarity, let us simply compare our bound with that of Sadiev et al. Our bound can be expressed as $$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha - 1}} + \frac{(L_1R_0)^{2+\alpha}}{\delta}\right).$$
Bound from Sadiev et al. is $$K = \tilde{\mathcal{O}}\left(\frac{L_0(1 + L_1R_0\exp(L_1R_0))R_0^2}{\varepsilon} + \left(\frac{R_0\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha - 1}}\right).$$
Without loss of generality, let us assume $L_1R_0 >> 1$ for a consideration the case with the "problematic" term $\frac{1}{\delta}$. Thus, we need to compare $$((L_1R_0)^{\frac{\alpha}{\alpha-1}} - 1)\left(\frac{R_0\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha - 1}} + \frac{(L_1R_0)^{2+\alpha}}{\delta} \text{ vs } \frac{L_0(L_1R_0\exp(L_1R_0))R_0^2}{\varepsilon}.$$

It is very important to clarify that, in general, it is difficult to interpret a comparison of the bounds directly from the above formula, since many parameters are involved. However, one can consider a specific choice of these parameters in order to examine a numerical comparison to demonstrate how severe these bounds may differ. For instance, let us take $\alpha = 3/2$, $\sigma = 1$, $L_0 = 1$, $L_1 = 10$, $R_0 = 10$, $\varepsilon = 10^{-10}$ (which is a really good choice for the accuracy). If we begin comparing the bounds in the order of the terms, we obtain

$$10^{36} + \frac{10^7}{\delta} \text{ vs } 2.5 \cdot 10^{57}.$$

As a direct result, our bound will be *worse* than the bound from Sadiev et al. in the case

$$\delta \lesssim 0.4 \cdot 10^{-50}.$$
This failure level is **negligible from a practical point of view**. And, in fact, it is quite natural: the factor $\frac{1}{\delta}$ becomes dominant compared to Sadiev et al. if it is approximately higher than $\exp(L_1R_0)$. And this is precisely the most important contribution of our work: the above example clearly demonstrates ***why one should avoid this exponential factor in upper bounds***, which is *one of the central questions* addressed in our study: how to derive a high-probability convergence for the convex case for generalized smoothness and heavy-tailed stochasticity *without exponential depencency*. And, indeed, our paper is the first who addressed this problem.

***Nature of the stochasticity.*** If one considers all results for the convex case under generalized smoothness without using probability amplification, then, besides our work, there are two papers that can be compared in terms of the additional term:

-- Deterministic (Gorbunov et al. (2024)): ${\mathcal{O}}\left((L_1R_0)^2\right);$

-- Light-tailed noise (Gaash et al. (2025)): ${\mathcal{O}}\left((L_1R_0)^2\log\left(\frac{1}{\delta}\right)\right);$

-- Heavy-tailed noise (our paper): ${\mathcal{O}}\left(\frac{(L_1R_0)^2}{\delta}\right)$
(here we decrease the power $2 + \alpha \rightarrow 2$ since we believe that it is the artifact of the proof). At present, ***neither the lower bounds nor even the in-expectation upper bounds*** are known. Therefore, term $\frac{1}{\delta}$ may in fact be *unimprovable*: it may be caused by the very nature of the stochasticity rather than by an imprecise analysis. Of course, this raises the question of why probability amplification yields the desired result, whereas our analysis produces a “non-classical” bound. Unfortunately, comparing them directly is **simply incorrect**: applying probability amplification changes the convergence criterion (a minimum over all runs appears), whereas in the above-mentioned works and in our paper the convergence criterion for a single run is *last-iterate* for the first one and *ergodic* over a "good" set for the last two results.

In fact, the question remains whether one can obtain a "classical" high-probability bound for the convex $(L_0, L_1)$-smooth case with heavy-tailed stochastic noise while ***avoiding an exponential dependence on $R_0$***. This question is still open, but we believe that our study, **being the first to address this problem**, can serve as a starting point for future work that may answer it.

>***Classical $(L_0, L_1)$-smoothness.***

We hope that we fully addressed this concern since the assumption considered in our paper is relevant and standart, according to recent papers.

Additionally, we want to mention that we have prepared a nonconvex analysis (which, as expected, turned out to be easier than the convex case - see the rebuttal comment to the Reviewer vNTE).

We would like to thank you for your comments. We hope that our response addresses your remaining concerns about our work and will allow you to reconsider the final evaluation.





# Short version starts

## Reviewer vNTE

Thank you for the careful reading and the positive assessment of the paper. We appreciate your comments and will revise the paper accordingly.

---

> On convexity vs. practice / the nonconvex setting

Thank you for mentioning this point. Indeed, one of the main motivations for our study was the consideration of generalized smoothness and heavy-tailed stochasticity, both of which naturally arise in the analysis of deep learning models. According to this, we respectfully agree that investigating the non-convex case is essential for description of the real-world scenario and can be considered as a natural next step. However, there are some reasons why we chose to focus on the convex setting instead:

- **Relevance of convex analysis**. Recent studies emphasize the relevance of convex analysis in modern research. For instance, works [1], [2], and [3] show that deep neural networks often exhibit locally convex-like behavior in certain regions of the parameter space. This observation suggests that convergence analyses developed for convex functions can, to some extent, be applied to modern deep learning problems. Furthermore, studies such as [4] demonstrate that convex optimization theory aligns well with large-scale model training, particularly regarding learning-rate schedules.

- **Hidden exponentially large factors**. One of the motivations of our work is to obtain convergence guarantees without any exponentially large involving $L_1 R_0$, neither explicit or implicit (here $R_0 = \|\| x_0 - x^* \|\|$). However, in the non-convex setting, all existing convergence or complexity bounds that we are aware of explicitly depend on $\Delta := f(x_0) - f^*$, which can be exponentially larger than $R_0$ when $x^*$ exists. As a toy example, one can consider $f(x) = \|\| x \|\|^4$ and $\|\| x_0 \|\| = 100$; we have $R_0 = 100$, while $\Delta = 10^8$, which is much larger than $R_0$. The situation becomes even worse for functions $f(x)$ that grow exponentially fast as $\|\|x - x^* \|\| \to \infty$. That is, a straightforward application of non-convex results to the convex case leads to highly suboptimal guarantees.

Therefore, since the convex setting is both theoretically important and still practically relevant, we decided to focus our analysis on the convex scenario.

---

> Missing related work / condition on $K$

Thank you for pointing out [5]. We agree it is relevant and will add it to Related Work. Since it studies a nonconvex setting, we will discuss it as complementary rather than directly comparable. We also agree that the current self-referential condition on $K$ is awkward; in the revision we will replace it with a simpler sufficient statement such as $K=\widetilde\Omega((L_1R_0)^{2+\alpha}/\delta)$.

---

> Dependence on $\delta$

We agree this deserves a clearer explanation. In the current theorem, the $1/\delta$ term appears only in the large-radius regime and only as a lower bound on the horizon. However, one can remove this dependence from the *final* confidence parameter by standard amplification: run the method with any fixed constant failure level $\delta_0<1/4$ (for example, $\delta_0=1/8$), so that one run succeeds with constant probability, and repeat independently $O(\log(1/\beta))$ times. Combined with the implementable selection rule of Appendix C (and a robust comparison across the $O(\log(1/\beta))$ outputs), this yields total complexity

$$
\widetilde O\!\left(
\frac{L_0R_0^2}{\varepsilon}
+
\left(\frac{\max\{1,L_1R_0\}R_0\sigma}{\varepsilon}\right)^{\!\frac{\alpha}{\alpha-1}}
+
(L_1R_0)^{2+\alpha}
\right)\log\frac1\beta,
$$

so the dependence on the target confidence becomes logarithmic. We will add this corollary explicitly.

---

> Main challenge for a nonconvex extension

At a high level, the difficulty is exactly the mismatch above: generalized smoothness makes the nonconvex scale $\Delta_0$ and the convex scale $R_0$ behave very differently, so the proof techniques and the resulting guarantees do not transfer in a straightforward way. We agree that closing this gap to nonconvex practice is important, and we will state this more clearly.

---

**Thank you again for the constructive feedback. If these clarifications address your concerns, we would be grateful if you would consider updating your score.**

---

*References*

[1] Kleinberg et al., An alternative view: When does SGD escape local minima? (ICML 2018)
[2] Zhou et al., Sgd converges to global minimum in deep learning via star-convex path (ICLR 2019)
[3] Liu et al., Loss landscapes and optimization in over-parameterized non-linear systems and neural networks (ACHA 2022)
[4] Schaipp et al., The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training (ICML 2025)
[5] Armacki et al., Optimal High-probability Convergence of Nonlinear SGD under Heavy-tailed Noise via Symmetrization (ArXiv preprint, 2025)

## Reviewer Uqxp

Dear Reviewer,

Thank you for the thoughtful and constructive review. We are glad you found the technical analysis careful, and we will revise the paper along the lines you suggest.

> Origin of the two regimes $4R_0 \lessgtr 1/L_1$

This split is driven by the interaction between generalized smoothness and heavy-tail control. If $4R_0\le 1/L_1$, then on the high-probability invariant ball around $x^\star$, generalized smoothness effectively reduces to local $L$-smoothness with $L=O(L_0)$; this is why the proof becomes close to Sadiev et al. In contrast, when $4R_0\ge 1/L_1$, generalized smoothness remains active and we must separately control iterations with $\|\nabla f(x_k)\|>\lambda/2$ (our set $T_3$). So the distinction is not cosmetic: it reflects whether the iterates stay in a region where the $L_1$-term can be absorbed into a local smoothness constant. The split disappears when $L_1=0$.

> Complexity comparison / Table 1

We agree completely. In the revision we will rewrite Table 1 in terms of target accuracy $\varepsilon$ and make the dependence on the confidence parameter explicit, which should make the comparison to prior high-probability results much easier to read.

> Origin of the extra $1/\delta$ term

This term comes specifically from the large-radius regime. There, the proof must control the rare iterations in $T_3=\{k:\|\nabla f(x_k)\|>\lambda/2\}$. On the good event, the number of such iterations is at most $C_1=\Theta((L_1R_0)^2\log(K/\delta))$, but each one is handled only via Markov-type control under finite $\alpha$-moment noise. This is the step that introduces the extra horizon requirement. It does not appear in the small-radius regime or when $L_1=0$.

> Probability amplification and final iterate selection

We also agree that a cleaner confidence statement is desirable. One can take any fixed constant $\delta_0<1/4$ (e.g., $1/8$), run the algorithm once to get constant success probability, and repeat independently $O(\log(1/\beta))$ times. This removes the inverse dependence on the *final* confidence parameter and gives total complexity

$$
\widetilde O\!\left(
\frac{L_0R_0^2}{\varepsilon}
+
\left(\frac{\max\{1,L_1R_0\}R_0\sigma}{\varepsilon}\right)^{\!\frac{\alpha}{\alpha-1}}
+
(L_1R_0)^{2+\alpha}
\right)\log\frac1\beta.
$$

Appendix C is compatible with this: first select one implementable iterate per run exactly as in Appendix C, then robustly compare the $O(\log(1/\beta))$ final candidates. We will add this discussion and a short intuitive paragraph explaining the meaning of the two regimes.

Thank you again for the detailed questions. We believe your suggestions will materially improve the paper, and we hope the clarifications above address your concerns.

## Reviewer xNrk

Dear Reviewer,

Thank you for the careful review and for highlighting both the strengths and the weaknesses. We agree the paper can present the contribution more sharply, and we will revise accordingly.

> Experimental evaluation

Our experiments were meant as targeted validation of the two paper-specific claims: (i) there is a genuine trade-off between the clipping regime needed for generalized smoothness and the one needed for heavy-tailed noise, and (ii) the independent-sampling variant is much less robust under heavy tails. Since the paper is primarily theoretical, we used a controlled synthetic setting to isolate these effects rather than to benchmark Clip-SGD on a large application. That said, we agree the presentation should make this intent clearer, and we will emphasize that the experiments are illustrative rather than exhaustive. We will also move a concise experimental summary into the main text. If space permits, we will also add a small rate-oriented plot or, at minimum, tone down the corresponding claims.

> Presentation / balance of the paper

This is helpful feedback. We will shorten redundant discussion, move some material out of the main text, and make the main theorem and its takeaways more prominent. In particular, we will explicitly state earlier that the key novelty is the large-radius regime, while the small-radius regime mainly serves to recover the correct boundary/special case.

> Scope / novelty

We respectfully believe the main contribution is stronger than it may currently appear. The core new part is not Regime 1 but Regime 2, where generalized smoothness and heavy-tailed noise conflict: clipping must both control gradient norms for the $(L_0,L_1)$-smooth analysis and suppress rare extreme stochastic gradients. The proof therefore requires a new decomposition into $T_1,T_2,T_3$ and a different induction from prior high-probability analyses of Clip-SGD. To the best of our knowledge, this is the first convex high-probability analysis in the joint setting, and it avoids the exponential $L_1R_0$ factors that arise from reducing to local $L$-smoothness. We will highlight this more clearly. As an additional clarification, the current $1/\delta$ horizon term can be converted into logarithmic dependence on the final confidence by a standard restart/probability-amplification corollary, which we will add in the revision.

> Assumption 2 notation

Thank you for catching this. We will define it explicitly as

$$
[x,y]=\{tx+(1-t)y:\ t\in[0,1]\},
$$

so $\sup_{u\in[x,y]}$ is the supremum along the line segment joining $x$ and $y$.

Thank you again for the constructive feedback. If the revised framing addresses your concerns, we would be grateful if you would reconsider the score.

## Reviewer mLLq

Dear Reviewer,

Thank you for the careful review and for acknowledging that the paper is technically sound. We agree that the current presentation does not highlight the main contribution sharply enough, and we appreciate the chance to clarify it.

> Actual contribution / Regime 1 vs. Regime 2

We agree that the small-radius regime is close to prior work once generalized smoothness can be absorbed into local $L$-smoothness. The core contribution is the large-radius regime $4R_0\ge 1/L_1$: here the two roles of clipping genuinely conflict, and prior arguments for heavy-tailed Clip-SGD or deterministic $(L_0,L_1)$-smooth optimization do not directly apply. Our proof introduces the $T_1/T_2/T_3$ decomposition and a new induction that tracks the rare large-gradient steps separately. This is the part we view as conceptually new, and we will make that emphasis much clearer in the revision.

> Is the result really “high probability” given the $1/\delta$ horizon?

We agree that the current statement is non-standard and should be complemented by a cleaner corollary. The key point is that the inverse dependence on $\delta$ is only needed to obtain one-run success at confidence $1-\delta$ in the large-radius regime. For an end-to-end guarantee, one can fix any constant $\delta_0<1/4$ (e.g., $1/8$), obtain constant success probability in one run, and then apply standard probability amplification with $O(\log(1/\beta))$ independent restarts. Combined with the implementable selection rule from Appendix C (first choose one output per run, then robustly compare the $O(\log(1/\beta))$ outputs), this yields total complexity

$$
\widetilde O\!\left(
\frac{L_0R_0^2}{\varepsilon}
+
\left(\frac{\max\{1,L_1R_0\}R_0\sigma}{\varepsilon}\right)^{\!\frac{\alpha}{\alpha-1}}
+
(L_1R_0)^{2+\alpha}
\right)\log\frac1\beta,
$$

i.e., logarithmic dependence on the final confidence level. We will add this corollary explicitly. So we agree that the current theorem statement alone is not the cleanest possible presentation; our point is that the analysis already implies the standard logarithmic-confidence guarantee after a standard amplification step.

> Why the nonconvex case is challenging

We agree this is an important open direction. Under classical $L$-smoothness, convex and nonconvex rates are closely related because $\Delta_0=f(x_0)-f^\star$ can be upper bounded by $O(LR_0^2)$. Under generalized smoothness, this correspondence breaks: nonconvex results naturally scale with $L_0\Delta_0$, while convex results scale with $L_0R_0^2$, and $\Delta_0$ can be exponentially larger than $R_0^2$. This is why a nonconvex extension is not a routine weakening of our assumptions but a structurally different problem.

> Missing citation / definition of $(L_0,L_1)$-smoothness

Thank you for pointing us to An et al.; we will add and discuss it in Related Work. On the definition, we may simply not have been clear enough. The segment-based form

$$
\|\nabla f(x)-\nabla f(y)\|\le\Big(L_0+L_1\sup_{u\in[x,y]}\|\nabla f(u)\|\Big)\|x-y\|
$$

is standard in recent convex generalized-smoothness papers. Some nonconvex works use a local one-step form tailored to proofs based on $x_{k+1}-x_k$; the two are related, but the segment form is the natural one for our convex analysis because the proof sometimes compares $x_k$ directly with $x^\star$. We will clarify this distinction and discuss the relation to the cited nonconvex paper more carefully.

Thank you again for the thoughtful comments. We hope these clarifications address the main concerns and would be grateful if you would reconsider the score.



# Short version ends


# Reviewer vNTE (score 4, confidence 4)

Dear Reviewer,

Thank you for your time and valuable feedback. We appreciate that you highlight the strengths of our paper. We are also open to further discussion regarding any concerns about our work, and we would be happy to answer any questions you may still have.

> ***W1:*** There is a slight discrepancy between the authors' assumptions and practice, in the sense that...

***A:*** Thank you for mentioning this point. Indeed, one of the main motivations for our study was the consideration of generalized smoothness and heavy-tailed stochasticity, both of which naturally arise in the analysis of deep learning models. According to this, we respectfully agree that investigating the non-convex case is essential for description of the real-world scenario and can be considered as a natural next step. However, there are some reasons why we chose to focus on the convex setting instead:

- **Lower degree of prior exploration**. Compared to the convex case, generalized smoothness appears to be significantly better studied in the non-convex setting, even under heavy-tailed noise. There are a lot of paper, which investigate the application of normalization or sign-based approaches. At the same time, convex analysis plays an important role in optimization theory since a lot of techniques, which originally emerged from the convex setting, continue to be popular and widely used even for non-convex problems. There are quite examples - from momentum techniques to local-update methods in distributed optimization.

- **Relevance of convex analysis**. Recent studies emphasize the relevance of convex analysis in modern research. For instance, works [1], [2], and [3] show that deep neural networks often exhibit locally convex-like behavior in certain regions of the parameter space. This observation suggests that convergence analyses developed for convex functions can, to some extent, be applied to modern deep learning problems. Furthermore, studies such as [4] demonstrate that convex optimization theory aligns well with large-scale model training, particularly regarding learning-rate schedules.

- **Hidden exponentially large factors**. One of the motivations of our work is to obtain convergence guarantees without any exponentially large involving $L_1 R_0$, neither explicit or implicit (here $R_0 = \|\| x_0 - x^* \|\|$). However, in the non-convex setting, all existing convergence or complexity bounds that we are aware of explicitly depend on $\Delta := f(x_0) - f^*$, which can be exponentially larger than $R_0$ when $x^*$ exists. As a toy example, one can consider $f(x) = \|\| x \|\|^4$ and $\|\| x_0 \|\| = 100$; we have $R_0 = 100$, while $\Delta = 10^8$, which is much larger than $R_0$. The situation becomes even worse for functions $f(x)$ that grow exponentially fast as $\|\|x - x^* \|\| \to \infty$. That is, a straightforward application of non-convex results to the convex case leads to highly suboptimal guarantees.

Therefore, since the convex setting is both theoretically important and still practically relevant, we decided to focus our analysis on the convex scenario.

>***W2:*** A closely related work [5], that studies high-probability (HP) convergence of nonlinear SGD (including clipping) in the presence of heavy-tailed noise was omitted from the literature. When providing...

***A:*** Thank you for this clarification. We agree that work [5] is relevant, and we will add it to the Related Work section. It is important to clarify, however, that we cannot directly compare our results with those of [5], since that paper considers a non-convex setting, which differs substantially from the convex case (see our response to W1). We would also like to thank you for pointing out expression on $K$. We initially chose to leave it in that form because it is a transcendental equation, and obtaining an exact algebraic expression for $K$ is impossible. However, for the sake of simplicity, we will reformulate this condition on $K$.

> ***W3:*** In addition to the comment in the soundness, the significance of the paper is slightly decreased due to the fact that, while the authors study high-probability guarantees, they still show a polynomial dependence on the confidence level $\delta$, in the requirement on the number of iterations $K = \tilde{\Omega}(1/\delta)$. While the authors provide...


***A:*** Thank you for this very important clarification. Indeed, factor $1/\delta$ in the final bound is troubling - it is non-standard, and therefore, if we take $\delta$ to be extremely small, our bound would not even improve upon the result of Sadiev et al. However, we will explain how to eliminate this factor from the final bound, essentially recovering the same order as the light-tailed noise bound in Gaash et al. To achieve this, we need to use multiple runs of our algorithm. For each such run, we set $\delta$ equal to $1/2$, and then within a single run we can guarantee convergence with probability at least $1/2$. By performing $\log_2(1/\beta)$ runs, we guarantee that with probability at least $1 - \beta$, at least one of the runs will attain the desired accuracy $\varepsilon$. That is, now we have

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + \frac{(L_1R_0)^{2+\alpha}}{\delta}\right).$$

Substituting $\delta = 1/2$, we obtain the convergence to $\varepsilon$-accuracy with probability at least $1/2$ after

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)$$
iterations. With $\log_2(1/\beta)$ starts, we obtain the result similarly to probability amplification technique:

$$K = \tilde{\mathcal{O}}\left(\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)\log_2(1/\beta)\right)$$
iterations required to converge with probability at least $1 - \beta$. 

It is crucial to note that the methodology for selecting the final point, which we describe in Section C, is also applicable to the multiple-restart setting. Therefore, choosing the run that yields the best point is not a critical issue from the standpoint of practical implementation.

As a consequence, we eliminate the factor $1/\delta$ in the final bound.

>***Q1:*** Overall, I find the problem studied in this work highly relevant and appreciate the merits of the paper. At the same time, ...

***A:*** It is important to understand that generalized smoothness creates a substantial mismatch between the convex and non-convex cases. This arises precisely for one of the reasons we mentioned earlier: $\Delta_0$ contains a hidden exponential dependence on $R_0$. For this reason, even the deterministic bounds for these two cases cannot be placed on the same footing, since they correlate poorly with one another. It is enough to write out the expected parts of the convergence bounds with respect to $L_0$:

-- $\mathcal{O}(\frac{L_0\Delta_0}{K})$ for non-convex case;

-- $\mathcal{O}(\frac{L_0R_0^2}{K})$ for convex case.

Therefore, the expected results in the non-convex case are *conceptually different* from those in the convex case (since $\Delta_0 \sim \exp(R_0)$ in the worst case). As a consequence, the technical aspects of the proof construction are different as well. We believe that the non-convex scenario is a challenging and very important problem, but it seems that its analysis differs *severely*. Hence, given the relevance of our setting (as discussed earlier in ***W1***), we have not considered the non-convex case so far and cannot say what technical subtleties would arise there.

We hope that we have addressed all the concerns you raised. If you still have any questions, we would be very happy to discuss them. If you feel that we have resolved the concerns and shortcomings, we kindly ask you to reconsider your evaluation of our work.

---

*References*

[1] Kleinberg et al., An alternative view: When does SGD escape local minima? (ICML 2018)
[2] Zhou et al., Sgd converges to global minimum in deep learning via star-convex path (ICLR 2019)
[3] Liu et al., Loss landscapes and optimization in over-parameterized non-linear systems and neural networks (ACHA 2022)
[4] Schaipp et al., The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training (ICML 2025)
[5] Armacki et al., Optimal High-probability Convergence of Nonlinear SGD under Heavy-tailed Noise via Symmetrization (ArXiv preprint, 2025)



(ТУТ ЖЕСТКИЙ ПЕРЕБОР ПО СИМВОЛАМ, сейчас ОЧЕНЬ МНОГО...)

# Reviewer Uqxp (score 4, confidence 3)

Dear Reviewer,

Thank you for your time and valuable feedback. We are grateful for your recognition of the strengths of our work and for your suggestions on how to improve it. We are also open to further discussion of your remaining concerns after our responce.

>***Suggestion 1: description of ratio between $4R_0$ and $1/L_1$***.

***A:*** Thank you for this clarification. We will propose additional explanation for this distinction. In fact, that phenomenon appear from the idea of the “competition” between the clipping parameter for generalized smoothness ($\lambda \sim L_0/L_1$) and for the $L$-smooth setting in the presence of heavy-tailed noise ($\lambda \sim LR_0$). If the initial distance is small, then, due to the inductive nature of the proof, we can say that $L$-smoothness holds with some constant $L$. This is precisely why, in case $4R_0 \leq 1/L_1$, our proof is similar to Sadiev et al. 

>***Suggestion 2: removing the components for the comparison with previous results***

***A:*** We will add a comparison with existing results in that format.

>***Suggestion 3: presentation***

***A:*** Thank you for this comment. We understand that discussion of two regimes in Theorem 1 may look technical, but actually it is indeed technical. We believe that considering these cases is a technical way of addressing the problem at hand, which is why we intended this discussion to remain technical in nature. As for Table 1, we will revise it by expressing the iteration complexity in terms of the $\varepsilon$-accuracy. We provide a slightly more detailed response to this comment in Question 1.

>***Q1: Complexity comparison in Table 1***

***A:*** As we said before, we will add the comparison of existing bounds depending on $\varepsilon$-accuracy. We agree that it is more representative for the readers.

>***Q2, Q4 and Q5: Probability amplification vs. Markov’s inequality, Interpretation of the additional $1/\delta$ term and Final iterate selection.***

***A:*** Thank you for these questions. Indeed, if we apply a probability amplification scheme, then the resulting high-probability guarantee is actually *better* than ours. In other words, if the expectation bound exists in the form described in Section 5.2, then Markov’s inequality yields a strictly worse result than ours, whereas probability amplification yields a better one (in this comparison, we are implicitly suppressing the convergence criterion). Nevertheless, we are happy to provide a result based on the idea of probability amplification, showing how to: (i) derive the final convergence bound *without* factor $1/\delta$ (Q4); and (ii) explain why the selection scheme described in Section C of the Appendix remains unaffected (Q2, Q5). 

First, we have:

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + \frac{(L_1R_0)^{2+\alpha}}{\delta}\right)$$

iterations to reach $\varepsilon$-accuracy with probability at least $1 - \delta$. Substituting $\delta = 1/2$, we obtain the convergence with probability at least $1/2$ after

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)$$
iterations. Now, let us set $\log_2(1/\beta)$ runs of the algorithm (similarly to probability amplification); therefore, the probability of having at least one run which reachs $\varepsilon$-accuracy is greater than $1 - \beta$. Moreover, the total number of iterations can be expressed as

$$K = \tilde{\mathcal{O}}\left(\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)\log_2(1/\beta)\right)$$
iterations required to converge with probability at least $1 - \beta$,

which coincide with the classical result (containing only $\log(1/\beta)$).

It is crucial to note that the methodology for selecting the final point, is also applicable to the multiple-restart setting (as in probability amplification). Section C describes how to choose the iterate in *one* run. However, we *cannot* apply our idea from this section in a ***direct form*** to the multi-start approach since the convergence theorem (Theorem 1) provides an ergodic convergence for "good" iterares. At the same time, a final criterion with probability amplification looks as follows:

$$\min_{runs=1, \ldots, \log(1/\beta)} \text{Criterion for one run} = \tilde{\mathcal{O}}(\ldots).$$

Therefore, it is enough to compare the outputs (output is feasible for us due to the construction of selection scheme) for these runs and find the best one. It requires $\mathcal{O}(\log(1/\beta))$ selection schemes, which looks feasible from the viev of practicality.

It is worth noting that we will add this decription in the formal way in the final version of the paper. We also appreciate you for raising these important points.

>***Q3: Origin of the two regimes in Theorem 1***

***A:*** We believe that we partially addressed this question in ***W1***. In addition, we can clarify that the presence of these regimes will be annihilated if someone is assuming $L_1 = 0$, i.e. $L_0$-smooth functions are considered.


We hope that we have addressed all of your concerns regarding our work. We would be very happy to discuss any remaining questions, and we will certainly take your suggestions for improvement into account. If possible, we respectfully ask you to reconsider your evaluation of the paper if we have resolved your concerns.

(ТУТ ПЕРЕБОР НА 550 СИМВОЛОВ)


# Reviewer xNrk (score 3, confidence 3)

Dear Reviewer,

Thank you for your time and feedback. We appreciate that you highlight the strengths of our paper. We are also open to further discussion regarding any remaining concerns about our work.

>***W1:*** The experimental evaluation is relatively limited. The experiments appear mainly illustrative and are conducted on synthetic settings, and the experiments do not explicitly verify the theoretical convergence rates predicted by the analysis (e.g., the dependence on $K^{\frac{\alpha-1}{\alpha}}$)

***A:*** Thank you for this comment. In essence, the Clip-SGD method has already been applied many times in various papers to practical tasks such as LLM fine-tuning, so constructing the same type of experiments would be somewhat unusual in the context of our work. We conducted numerical experiments to address the two main claims made in our paper: (i) whether there truly exist threshold regimes for gradient clipping in which it handles either generalized smoothness or heavy-tailed noise, and (ii) whether the double-sampling strategy from Gaash et al. works in the presence of heavy-tailed noise. Accordingly, we show that even on a simple problem, both of these phenomena do indeed arise.

In addition, we would like to emphasize that our work is primarily theoretical in nature - we provide the *first high-probability bounds for the setting under consideration*. To the best of our knowledge, there is no existing literature that studies this setting under the same assumptions.

>***W2: Presentation***

***A:*** Thank you for your suggestion on improving the presentation of our work. We understand that the discussion of the results takes up a considerable amount of space, which may be concerning. The main reason for this is that this setting has not been studied before. This means that we need to: (i) interpret our convergence bounds; (ii) compare them with existing and even currently unavailable results; and (iii) compare our result, both experimentally and theoretically, with the only work that is close in setting (Gaash et al.). 

This is unavoidable - the paper is ***the first in its class***. That is why we devote so much space to discussing these results. 

As for the experimental part, we hope that we have addressed this concern in ***W1***.

Additionally, we would like to point out that in the final version of the paper, we will provide an *improved convergence bound* (we kindly ask you, if possible, to see our response to Reviewer Uqxp regarding $1/\delta$, Question 4).

>***W3:*** The technical contribution appears relatively limited in scope. 

***A:*** We kindly disagree with this concern. Our work is the first in this class of problems; the closest work in terms of setting is Gaash et al., to which we devoted Section 5.1 for a detailed comparison. Moreover, to the best of our knowledge, there is no existing analysis of Clip-SGD under these assumptions, even though generalized smoothness, heavy-tailed stochastic noise, and convexity are all assumptions that align well with the current state of the field. In addition, the proof technique we provide (see Section 4) has not appeared previously, due to, at least, the significantly more involved inductive step.

>***W4:*** Some components of the proof build heavily on existing work, which slightly reduces conceptual novelty.

***A:*** Thank you for this comment. We have been fully transparent in stating that Regime 1 is closely related to the work of Sadiev et al. However, the key contribution of our paper is the analysis of Regime 2; to the best of our knowledge, such an analysis has not appeared before. Therefore, it would not be accurate to say that the most important part of our contribution relies heavily on existing approaches.

>***Q1:*** In Assumption 2, for vectors $u, x, y \in \mathbb{R}^d$, please clarify the definition of $\sup_{u \in [x, y]}$

***A:*** This denotes the supremum over the interval $[x, y] = \{u \in \mathbb{R}^d| \ \exists t \in [0, 1]: u = tx + (1-t)y \}$.

We hope that we have fully addressed the concerns regarding our work. If you still have any remaining questions, we would be glad to answer them. We also kindly ask you to reconsider your evaluation of our work, if possible. 

# Reviewer mLLq (score 3, confidence 4) 

Dear Reviewer,

Thank you for your time and feedback. We appreciate for emphasizing the strengths of our paper. We also open for the further discuccion of the remaining questions after our responce.

>***W1:*** Even though the paper is sound, there is a big question about the actual contribution of the paper.
The first case, ..., the convergence is straightforward following from prior work/techniques, e.g Sadiev et al. So the technical contribution in this case is limited.
The second case ...


***A:*** Thank you for this important clarification. We agree that factor $1/\delta$ is somewhat troubling and may raise concerns about the practical applicability of this result. With this in mind, we will show how we plan to improve our bound in the final version of the paper. Applying the bound from our paper, we obtain that, in order to guarantee convergence with $\varepsilon$-accuracy with probability at least $1-\delta$, it is sufficient to make $K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha - 1}} + \frac{(L_1R_0)^2}{\delta}\right)$ iterations. Let us now substitute into this bound $\delta = 1/2$. Then, with this result in hand, we can apply the multiple-start technique to guarantee convergence with probability $1-\beta$, provided that the number of restarts is equal to $\log_2(1/\beta)$. Overall, this yields a final bound on total number of iterations with factor $\log(1/\beta)$, as desired. To be more presice, we have

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + \frac{(L_1R_0)^{2+\alpha}}{\delta}\right)$$

iterations to reach $\varepsilon$-accuracy with probability at least $1 - \delta$. Substituting $\delta = 1/2$, we obtain the convergence with probability at least $1/2$ after

$$K = \tilde{\mathcal{O}}\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)$$
iterations. If we set $\log_2(1/\beta)$ runs of the algorithm (similarly to probability amplification), the probability of having at least one run which reachs $\varepsilon$-accuracy is greater than $1 - \beta$ with the total number of iterations:

$$K = \tilde{\mathcal{O}}\left(\left(\frac{L_0R_0^2}{\varepsilon} + \left(\frac{L_1R_0^2\sigma}{\varepsilon}\right)^{\frac{\alpha}{\alpha-1}} + (L_1R_0)^{2+\alpha}\right)\log_2(1/\beta)\right)$$
iterations required to converge with probability at least $1 - \beta$,

This coincide with the desired result (containing only $\log(1/\beta)$). We will add this idea in the formal way in the final version of the paper.

>***Q1:*** Could you point out why it's challenging to handle the nonconvex case. It seems to me that even in the light tailed noise case, results for high-probability convergence for nonconvex, $(L_0, L_1)$-smooth functions are sparse? 

***A:*** We fully agree that the non-convex setting is also a serious, challenging, and interesting problem at present. However, it is important to understand that, compared with the classical $L$-smoothness [1], generalized smoothness makes the distinction between the non-convex and convex cases critical. To see this, it is enough to look at the kinds of results that are expected when studying the non-convex and convex settings. For instance, for $L$-smooth setting (without any acceleration) we expect to derive

-- $\mathcal{O}(\frac{L\Delta_0}{K})$ for non-convex case;

-- $\mathcal{O}(\frac{LR_0^2}{K})$ for convex case;

This upper bounds correlate well since $\Delta_0$ can be upper bounded as $\mathcal{O}(LR_0^2)$ -- this means that dependence of convergence bounds on $R_0^2$ for both cases are similar. At the same time, for generalized smoothness, we expect to reinstate deterministic term related to $L_0$:

-- $\mathcal{O}(\frac{L_0\Delta_0}{K})$ for non-convex case;

-- $\mathcal{O}(\frac{L_0R_0^2}{K})$ for convex case;

However, $\Delta_0$ can be exponentially large depending on $R_0$: it is enough to consider a simple example: $f(x) = ||x||^{2n}$. As a consequence, the non-convex and convex cases under generalized smoothness differ too substantially to be treated on the same footing. For this reason, the challenges that arise in analyzing the non-convex and convex setups are also different. In our work, we addressed the convex case, while fully recognizing that the non-convex case is structurally different in its analysis under generalized smoothness, since it implicitly hides an exponential dependence on parameter $R_0$. That is precisely why we cannot comment on the technical subtleties of that setting: we considered a setup that is conceptually different from the non-convex one *in terms of the convergence guarantees* that can be obtained. 

>***Q2:*** The following paper might be relevant which was not cited: Weixin An, Yuanyuan Liu, Fanhua Shang, Han Yu, Junkang Liu, Hongying Liu, Tight High-Probability Bounds for Nonconvex Heavy-Tailed Scenario under Weaker Assumptions, NeurIPS 2025.

***A:*** Thank you for this clarification. This work is indeed relevant, and we will certainly add it to the Related Work section.

>***Q3:*** It would be good if you could discuss the relevance of this paper, although this paper uses a non-standard definition of $(L_0, L_1)$-smoothness and it's unclear to me what the results actually are.

***A:*** We kindly disagree with the claim that the generalized smoothness assumption considered here is non-standard. It is consistent with recent works on generalized smoothness in the convex setting [2], [3]. Most likely, the assumption you regard as standard is the following inequality: $$||\nabla f(y) - \nabla f(y)|| \leq (L_0 + L_1||\nabla f(x)||)||y - x||$$
for all $x,y: ||x - y||\leq 1/L_1$. In most cases, this formulation is used precisely in the non-convex setting, since the proof strategy often involves taking $y = x_{k+1}$ and $x = x_k$; applying clipping or normalization to the gradient estimator then makes it possible to guarantee $||x_{k+1} - x_k||\leq 1/L_1$. In the convex case, however, the situation is somewhat different, because it is sometimes necessary to substitute the optimal point. Thus, we consider the assumption as in our paper; however, it is indeed standart setting for generalized smooothness.


We hope that we have fully addressed the concerns regarding our work. If you still have any remaining questions, we would be happy to answer them. We also kindly ask you to reconsider your evaluation of our work, if possible, if all concerns have been resolved.

---

*References*

[1] Y. Nesterov, Introductory lectures on convex optimization: A basic course, 2013.
[2] Gaash et al., Convergence of Clipped SGD on Convex $(L_0,L_1)$-Smooth Functions, NeurIPS 2025
[3] Gorbunov et al., Methods for convex $(L_0, L_1)$-smooth optimization: Clipping, acceleration, and adaptivity. ICLR 2025
