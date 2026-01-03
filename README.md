# VulBlurrer

This repository contains the code for the paper Black-box Adversarial Attacks against Pre-trained Vulnerability Detection Models

## 1. Overview

Pretrained code models have demonstrated strong capabilities in code understanding and analysis, becoming important tools and research focuses in the field of source code vulnerability detection. However, similar to traditional deep learning models, pretrained code models exhibit vulnerabilities in robustness when exposed to carefully crafted adversarial code. Attackers can mislead the model into classifying vulnerable code as non-vulnerable by introducing semantically preserving perturbations, posing significant threats to software security. Therefore, adversarial attacks against pretrained vulnerability detection models not only serves as an effective approach to evaluate the robustness of pretrained code models but also provides critical insights for the development of future vulnerability detection models and defense mechanisms. In the hard-label black-box attack scenario, we propose VulBlurrer, a black-box adversarial attack method targeting pretrained vulnerability detection models. VulBlurrer and baseline methods are evaluated on pretrained vulnerability detection models including CodeBERT, GraphCodeBERT, CodeT5 and UniXcoder. 

## 2. Repository Structure

```text
VulBlurrer/
├── BigVul/        # Dataset
├── CodeTAE/       # baseline
├── NonVulGen/     # baseline
├── VulBlurrer/    # The proposed VulBlurrer framework
├── README.md
```

- **BigVul/**
- **CodeTAE/**
- **NonVulGen/**
- **VulBlurrer/**

## 3. Acknowledgement

We are very grateful that the authors of CodeBERT, GraphCodeBERT, CodeT5, UniXcoder, CodeTAE and NonVulGen make their code publicly available so that we can build this repository on top of their code.
