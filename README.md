# The Curious Case of SignMuon: Sign Before or After the LMO?
~~SignMuon: fast as Muon, communication-effective as SignSGD~~

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Maria Smirnova </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Alexey Kravatskiy </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main_ru.pdf)
- [Slides](slides/Final_talk_Smirnova.pdf)
- [Poster](slides/Poster_SignMuon.pdf)

## Abstract

The SignSGD gradient compression algorithm enables up to a $32\times$ reduction in communication volume, which is critical for federated learning in bandwidth-constrained environments. It often lags behind modern optimizers like Muon, which leverage the matrix structure of parameters to achieve superior convergence and performance. We propose SignMuon, an algorithm that applies sign compression to the Linear Minimization Oracle (LMO) update of the Muon optimizer, i.e., it computes `sign(LMO(·))`. However, SignMuon offers no global convergence guarantee: we construct a linear function on which SignMuon provably diverges. Based on this observation, we introduce EF-USignMuon, an Error-Feedback variant that compresses the residual with `sign(·)` and applies the Muon LMO on the server, computing `LMO(sign(·))` while keeping the uplink at one bit per parameter. It restores the descent property and converges on the same counterexample. We validate both methods on a synthetic convex problem and on centralized and federated CIFAR-10 classification.

## Code
### Centralized setting
The example of running code in the centralized setting:
```
python3 -m main --dataset cifar10 --optimizer signmuon --data data --device cuda:1 --epochs 50 
```
### Federated setting
The example of running code in the federated setting:
```
python3 -m federated_main --model cnn2 --dataset cifar10 --algorithm signmuon --rounds 2000 --n_parties 10 --n_steps 3 --batch_size 64 --device cuda:3 --eval_freq 100
```
