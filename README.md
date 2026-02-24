# SignMuon: fast as Muon, communication-effective as SignSGD

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
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Dmitry Kovalev, PhD </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

SignSGD is a popular algorithm to its strong performance in both centralized and decentralized settings:  it performs competitively with Adam while being communication-efficient, because it transmits up to 32x fewer bits than standard optimizers. This work proposes SingMuon, an algorithm that combines the structured LMO-based update of Muon with the sign compression of SignSGD. Preliminary experiments show that SignMuon achieves performance nearly on par with Muon in the centralized setting and outperforms signSGD in both centralized and federated settings. We establish theoretical convergence proofs for the smooth non-convex case, analyze modifications to improve stability, and investigate the broader class of SignA algorithms, where A is an LMO (linear minimization oracle)-based algorithm. Empirical validation is conducted on synthetic benchmarks, CIFAR-airbench, federated MNIST/CIFAR-10 classification, and NanoGPT training.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
