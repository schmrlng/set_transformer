# [Set Transformer](https://arxiv.org/abs/1810.00825)-based amortized clustering

The purpose of this repository is to demonstrate a basic ML workflow compatible with Stanford cluster computing resources. Specifically, this repo contains a reimplementation of Experiment 5.3 ("Amortized Clustering with Mixture of Gaussians") from
- [J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In *Proceedings of International Conference on Machine Learning*, 2019.](https://arxiv.org/abs/1810.00825)

https://user-images.githubusercontent.com/4130030/155441844-cc6d636c-cad7-4a9b-8e6a-8292b3c64a0d.mp4

## [Sherlock](https://www.sherlock.stanford.edu/) (Slurm) usage
0. **(Setup)** From the base directory of this repository, run
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip3 install -r requirements-jax.txt
   pip3 install -r requirements.txt
   ```
   If the last command fails, try [`pip3 install -r requirements.txt --no-cache-dir`](https://stackoverflow.com/questions/30550235/pip-install-killed).
1. **(Running)** Now run
   ```bash
   sbatch submit.sh
   ```
   to kick off a simple hyperparameter sweep as detailed in [`submit.sh`](submit.sh).
2. **(Monitoring)** To check the status of your jobs, run
   ```bash
   squeue -u $USER
   ```
   Once you can see that they've started, you should notice job log files (of the form `slurm-XXXXXXXX_X.out`) and a `checkpoints/` directory have been created. Observe the progress of your jobs from the command line with, e.g.,
   ```bash
   tail -f slurm*
   ```
   and/or navigate to [https://login.sherlock.stanford.edu/](https://login.sherlock.stanford.edu/pun/sys/dashboard/batch_connect/sys/sh_tensorboard/session_contexts/new) to run an [OnDemand TensorBoard session](https://www.sherlock.stanford.edu/docs/user-guide/ondemand/#tensorboard) (note that you will have to provide the relevant TensorBoard logdir, e.g., `$HOME/set_transformer/checkpoints`).

## General usage
[Install JAX](https://github.com/google/jax#installation) with your preferred accelerator support and then install the rest of the dependencies with `pip3 install -r requirements.txt`. Train from the command line with ```python3 main.py```; hyperparameters listed in [`config.py`](config.py) may be configured with [config flags](https://github.com/google/ml_collections/tree/master#config-flags), e.g.,
```bash
python3 main.py --config.input_encoding=sinusoidal --config.learning_rate=1e-3
```
In interactive workflows, you may choose to directly modify the [`ConfigDict`](https://github.com/google/ml_collections#configdict) returned by `config.get_config()` for usage with the functions in [`train_eval.py`](train_eval.py), e.g., [`train_eval.train_and_evaluate`](train_eval.py#L124).
