# Human review 2 — D4r2

Note ID: `UC752MnbRo`. Source: `signmuon_openreview_page.json`.

## Brief summary

The reviewer sees sufficient experiments to establish the phenomenon and rates the paper 7 (confidence 4). They require concrete writing fixes; larger experiments and a positive explanation would strengthen the contribution but are explicitly optional.

## Every concern / request

1. **R2.1 — Reviewer asks to identify sign as an LMO earlier.** Define the geometry in Section 2.

2. **R2.4 — Reviewer asks to distinguish raw signs from scaled EF messages.** Explain the EF scale when the variants are first introduced.

3. **R2.5 — Reviewer asks to distinguish centralized formulas from distributed algorithms.** Explain the N=1 correspondence and nonlinear aggregation.

4. **R2.6 — Reviewer questions the linear objective under lower boundedness.** Explain why the linear tests do not satisfy Assumption 1.

5. **R2.7 — Reviewer suggests consistent names for sign placements.** Make the composition order readable without changing identities.

6. **R2.2 — Reviewer suggests larger centralized and federated experiments.** Additional models and truly distributed language-model training would strengthen the evidence.

7. **R2.3 — Reviewer asks for progress on the theory-practice explanation.** Give a positive condition without claiming a complete explanation.

8. **R2.8 — Reviewer wants a realistic explanation of observed success.** Check the proposed sufficient condition on trained networks.

## Response priorities

Lead with the requested writing changes, including the naming decision. Defend why the linear examples are meaningful while supplying the new lower-bounded strengthening. Distinguish a conditional mathematical explanation from experimental verification.

## Original review

### title

Review

### review

The core problem considered in this paper is communication-efficient distributed training for structure-aware optimizers, by combining the recently proposed Muon matrix optimizer with the sign method, developing multiple variants depending on the ordering of the sign and LMO operations. The paper first theoretically shows that, in the matrix optimization setting, combining Muon and sign can result in an ascending (i.e., diverging) method, regardless of the ordering of the two operations. Turning to error-feedback for help, it is next demonstrated that SignMuon (sign after LMO) can not be fixed even with error-feedback. Extensive numerical experiments then demonstrate that SignMuon performs best in practice, (nearly) matching the performance of the original Muon, at a fraction of the communication cost. As such, the work highlights an important tension between the practical performance and theoretical guarantees of a communication-efficient version of Muon, providing some intuition and interpretation for the mismatch, without trying to resolve it.

### strengths

- **Important problem and insights**: The paper considers a very important and practically relevant problem of communication-efficient training, by combining the Muon optimizer with sign quantizer. More interestingly, the paper documents a very interesting tension between the theoretical guarantees and practical performance of a resulting variant dubbed SignMuon. 

- **Results**: While the insights that the paper provides are very interesting, they go significantly beyond just numerical observations. In particular, the counter-examples in Theorems 1-3  for which signed versions of Muon diverge are simple, instructive and yet very important to underline a theoretical basis for the claims made in this work. The numerical experiments then highlight the tension between theoretical guarantees and practical performance, demonstrating that theory potentially predicts a worst-case performance rather than a typical one (Appendix A.12), with consistent results across multiple settings (centralized, federated, nanoGPT). 

- **Writing**: The paper is generally very well-written and easy to follow, with a compelling narrative, starting from the abstract and introduction, all the way through to the theoretical and numerical results. The authors also provide important discussions, e.g., in Appendix A.7 on why their results are not contradicted by some concurrent works.

### weaknesses

- **Scale of numerical experiments**: While the numerical results are sufficient to illustrate the tension between practice and theory, additional experiments would help strengthen the argument even further. For example, adding some experiments in the federated setup, beyond CIFAR10 and ResNet18 would be very desirable, e.g., federated training of transformers and large language modes. Similarly, additional experiments in the centralized setup which further document that SignMuon does not diverge and performs on par with vanilla Muon would further strengthen the claim of a serious disconnect between theory and practice.

- **Lack of resolution**: Ideally, the paper would provide a resolution to the tension it has documented. However, this is not the case and it is honestly acknowledged throughout the paper, with the stated aim of informing the community of an interesting phenomena, rather than resolving it.

- **Writing**: There are a few places in the paper where the presentation is a bit unclear or causes some confusion. For example, it would be useful to point out in section 2, paragraph *"LMO optimizers"*, that sign is itself a LMO (this is mentioned in Section 4, but might be good to mention it earlier). Similarly, in the paragraph above, titled *"Sign compression"*, it is mentioned that sign sends "*one bit per coordinate, no index or scale sent beside it..."*, potentially giving the reader the impression that all sign-based methods (considered in this work) send only one bit per coordinate. However, that is clearly not the case with methods that incorporate error-feedback, where the sign compressor is modified to ensure it a contractive compressor and the scale of the input is also communicated with each update. This should be clarified in the main body, once the EF21 versions are mentioned. Next, in the paragraph below equation (6), the parameters transmitted on both uplink and downlink by Algorithms A2, A4 and A5  are outlined, however, it is not clear that this explanation applies to the said algorithms in the current form, as they seem to be centralized. The statement seems to apply for algorithm A8 with sign majority voting, however, that method is not the SignMuon or MuonSign as presented in the main body (e.g., equations (5)-(6)) and this should be clarified. In paragraph below equation (8), it is stated that "full precision Muon drive $f \rightarrow -\infty$", however, in light of Assumption 1, it might be more appropriate to have $f \rightarrow f^\star$. Finally, based on naming the "sign after Muon" method *SignMuon*, for the sake of consistency, it might be a good idea to rename the other two methods, namely "sign before Muon" and "sign before and after Muon" respectively as *MuonSign* (with the U dropped) and *SignMuonSign* (which can then be abbreviated in the text if needed, e.g., as SMS or similar).

### justification

The paper considers a very important problem of communication-efficient structure-aware learning by combining the matrix optimizer Muon and sign compression. The findings are very interesting, with a good narrative supported by strong theoretical and numerical results and this work raises some very interesting questions for the community. While the paper does not answer them directly, this is readily acknowledged. Instead, the paper is positioned as unveiling and extensively verifying a tension between theory and practice rather than resolving it, which itself is very important. In light of all of this, the final recommendation is an easy one to make.

### feedback

The third point listed in the weaknesses should be addressed. If the authors are able to address one of the first two points as well, it would make the contribution even stronger and the results more significant. However, the first two points are intended more as potential for further strengthening the claims and results, rather than severe criticisms or fundamental shortcomings.

### rating

7

### confidence

4
