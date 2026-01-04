
# VulBlurrer

This repository contains the code for the paper Black-box Adversarial Attacks against Pre-trained Vulnerability Detection Models

## 1. Overview

Pretrained code models have demonstrated strong capabilities in code understanding and analysis, becoming important tools and research focuses in the field of source code vulnerability detection. However, similar to traditional deep learning models, pretrained code models exhibit vulnerabilities in robustness when exposed to carefully crafted adversarial code. Attackers can mislead the model into classifying vulnerable code as non-vulnerable by introducing semantically preserving perturbations, posing significant threats to software security. Therefore, adversarial attacks against pretrained vulnerability detection models not only serves as an effective approach to evaluate the robustness of pretrained code models but also provides critical insights for the development of future vulnerability detection models and defense mechanisms. In the hard-label black-box attack scenario, we propose VulBlurrer, a black-box adversarial attack method targeting pretrained vulnerability detection models. VulBlurrer and baseline methods are evaluated on pretrained vulnerability detection models including CodeBERT, GraphCodeBERT, CodeT5 and UniXcoder.

## 2. Repository Structure

```text
VulBlurrer/
├── BigVul/        # Dataset
├── CodeTAE/       # Baseline
├── NonVulGen/     # Baseline
├── VulBlurrer/    # The proposed VulBlurrer framework
├── README.md
```

- **BigVul**

  **A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries**

  *Jiahao Fan, Yi Li, Shaohua Wang, Tien N. Nguyen.  In *Proceedings of the ACM/IEEE International Conference on Mining Software Repositories (MSR)*, 2020.*  [[paper](https://dl.acm.org/doi/pdf/10.1145/3379597.3387501)]

  Big-Vul is a high-quality real-world vulnerability dataset consisting of vulnerable and non-vulnerable functions extracted from open-source projects. The dataset covers publicly disclosed CVE vulnerabilities from 2002 to 2019, spanning 348 different open-source projects. In this work, we use the cleaned version and publicly available at: https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset.
- **CodeTAE**

  **Exploiting the adversarial example vulnerability of transfer learning of source code**

  *Yulong Yang, Haoran Fan, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen. IEEE Transactions on Information Forensics and Security, 2024, 19: 5880-5894.*  [[paper](https://ieeexplore.ieee.org/abstract/document/10531252)]

  CodeTAE is a cross-domain adversarial attack method designed for transfer learning–based source code classification models. Existing adversarial attack methods on source code often rely on strong assumptions, such as access to the victim domain dataset or query feedback from the target system, which are difficult or costly to obtain in real-world scenarios. To address this limitation, CodeTAE proposes a cross-domain attack threat model, where the adversary only has access to an open-sourced pre-trained code encoder. CodeTAE applies a variety of semantic-preserving code transformations and leverages a genetic algorithm to generate optimized identifiers, thereby improving the transferability of adversarial examples across domains and model architectures. The official implementation of CodeTAE is available at: https://github.com/yyl-github-1896/CodeTAE.
- **NonVulGen**

  **Black-box Adversarial Attack for Deep Vulnerability Detection Model**

  *Yubin Qu, Song Huang, Xiang Chen, Xingya Wang, Long Li, Dan Wang, Yongming Yao, Xiaolin Ju. Ruan Jian Xue Bao/Journal of Software (in Chinese).*  [[paper](https://www.jos.org.cn/jos/article/abstract/7379)]

  NonVulGen is an adversarial code generation framework targeting deep learning–based vulnerability detection models. Unlike prior studies that mainly focus on variable renaming, NonVulGen systematically investigates the effectiveness of introducing multiple semantic-preserving transformations to perturb source code. NonVulGen applies a diverse set of synonymous transformation operators and adopts a genetic algorithm–based optimization strategy to select transformation sequences with the highest fitness. This enables the generation of adversarial code samples that can successfully evade vulnerability detection while preserving program semantics. The official implementation of NonVulGen is available at: https://github.com/qyb156/RobustnessAttackVunerbilityDetection.

## 3. Acknowledgement

We are very grateful that the authors of [CodeBERT](https://arxiv.org/pdf/2002.08155), [GraphCodeBERT](https://openreview.net/pdf?id=jLoC4ez43PZ), [CodeT5](https://arxiv.org/pdf/2109.00859), [UniXcoder](https://arxiv.org/pdf/2203.03850), [CodeTAE](https://ieeexplore.ieee.org/abstract/document/10531252) and [NonVulGen](https://www.jos.org.cn/jos/article/abstract/7379) make their code publicly available so that we can build this repository on top of their code.
