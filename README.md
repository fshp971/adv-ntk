# Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach

This is the official repository for the ICLR 2024 paper ["Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach"](https://openreview.net/forum?id=1op5YGZu8X) by Shaopeng Fu and Di Wang.

## News

- 01/2024: The paper was accepted to [ICLR 2024](https://openreview.net/forum?id=1op5YGZu8X).
- 10/2023: The paper was released on [arXiv](https://arxiv.org/abs/2310.06112).

## Installation

### Requirements

- Python >= 3.10
- PyTorch == 2.0.1
- jax == 0.4.7 and jaxlib == 0.4.7 (`"jax[cuda11_cudnn82]"==0.4.7`)
- [neural_tangents](https://github.com/google/neural-tangents) == 0.6.2

### Build experiment environment via Docker

There are two ways to build the Docker experiment environment:

- **Build via Dockerfile**

  ```bash
  docker build --tag 'advntk' .
  ```

  Run the above command, and then the built image is `advntk:latest`.

- **Build manually**

  Firstly, pull the official PyTorch Docker image:

  ```bash
  docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
  ```

  Then, run the pulled Docker image and install following packages:

  ```bash
  pip install --upgrade "jax[cuda11_cudnn82]"==0.4.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  pip install neural-tangents==0.6.2

## Quick Start

The scripts for $\ell_\infty$-norm, $\rho=8/255$ experiments are collected in [./scripts](./scripts).	

**To run an experiment:** for example, execute the following command:

```bash
bash ./scripts/c10/mlp/advntk-r8.sh ./
```

**To use different perturbation radius $\rho$:** modify the following arguments accordingly:

```bash
--pgd-radius       # (float) adversarial perturbation radius
--pgd-steps        # (int) steps number in PGD
--pgd-step-size    # (float) step size in PGD
--save-dir         # (string) the path to the dictionary for saving experiment
```

## Citation

```
@inproceedings{fu2024theoretical,
  title={Theoretical Analysis of Robust Overfitting for Wide DNNs: An NTK Approach},
  author={Shaopeng Fu and Di Wang},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Acknowledgment

- Neural tangents: [https://github.com/google/neural-tangents](https://github.com/google/neural-tangents)
