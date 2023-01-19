import logging
import os
import random
import warnings

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from upcycle.logging.analysis import flatten_config
from upcycle.scripting import startup

import selfies as sf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lambo.models.gp_models import SingleTaskExactGP
from lambo.models.deep_ensemble import DeepEnsemble
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from torch.distributions import Normal


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    combine_dim = (
        -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    )
    return -pred_dist.log_prob(test_y) / test_y.shape[combine_dim]


@hydra.main(config_path="../hydra_config", config_name="regression", version_base=None)
def main(cfg):
    """
    general setup
    """
    random.seed(
        None
    )  # make sure random seed resets between Hydra multirun jobs for random job-name generation
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep="/")

    wandb.init(
        project=cfg.project_name,
        config=log_cfg,
        mode=cfg.wandb_mode,
        group=cfg.exp_name,
    )
    cfg["job_name"] = wandb.run.name
    cfg, _ = startup(cfg)  # random seed is fixed here

    dtype = torch.double if cfg.dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret_val = execute(cfg, dtype, device)
    except Exception as err:
        ret_val = float("NaN")
        logging.exception(err)

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


def execute(cfg, dtype, device):
    df_path = os.path.join(cfg.data_dir, cfg.task.df_path)
    df = pd.read_csv(df_path)

    selfies_data = list(map(sf.encoder, df[cfg.task.input_col]))
    selfies_alphabet = sf.get_alphabet_from_selfies(selfies_data)

    vocab_path = os.path.join(cfg.data_dir, "tmp_vocab.txt")
    with open(vocab_path, 'w') as f:
        for token in selfies_alphabet:
            f.write(token)
            f.write('\n')

    df.dropna(subset=cfg.task.label_cols, inplace=True)
    all_seqs = list(map(sf.encoder, df[cfg.task.input_col]))
    all_seqs = np.array([x for x in all_seqs]) 
    all_y = np.stack([df[col].values for col in cfg.task.label_cols], axis=-1)

    train_x, test_x, train_y, test_y = train_test_split(all_seqs, all_y, test_size=0.2, random_state=cfg.seed)
    if cfg.surrogate.holdout_ratio > 0.0:
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=cfg.surrogate.holdout_ratio)
    else:
        val_x, val_y = None, None

    y_scaler = StandardScaler()
    train_targets = y_scaler.fit_transform(train_y)
    if val_y is not None:
        val_targets = y_scaler.transform(val_y)
    else:
        val_targets = None
    test_targets = y_scaler.transform(test_y)

    # tokenizer = SELFIESTokenizer(dir_path=cfg.data_dir, selfies_vocab="tmp_vocab.txt")
    encoder = hydra.utils.instantiate(cfg.encoder)
    surrogate_model = hydra.utils.instantiate(cfg.surrogate, encoder=encoder)

    surrogate_model.fit(train_x, train_targets, val_x, val_targets, test_x, test_targets, encoder_obj="mlm", log_prefix=cfg.task.name)

    if isinstance(surrogate_model, SingleTaskExactGP):
        train_pred_dist = surrogate_model.predict_mvn(train_x)
        test_pred_dist = surrogate_model.predict_mvn(test_x)

        train_nlpd = negative_log_predictive_density(
            train_pred_dist,
            torch.from_numpy(train_targets.T).to(train_pred_dist.loc)
        ).item()
        test_nlpd = negative_log_predictive_density(
            test_pred_dist,
            torch.from_numpy(test_targets.T).to(test_pred_dist.loc)
        ).item()

        if val_y is not None:
            val_pred_dist = surrogate_model.predict_mvn(val_x)
            val_nlpd = negative_log_predictive_density(
                val_pred_dist,
                torch.from_numpy(val_targets.T).to(val_pred_dist.loc)
            ).item()
        else:
            val_nlpd = None
        
        train_targets_pred_mean = train_pred_dist.loc.t().cpu().numpy()
        test_targets_pred_mean = test_pred_dist.loc.t().cpu().numpy()

    elif isinstance(surrogate_model, DeepEnsemble):

        with torch.inference_mode():
            _, train_targets_pred_mean, train_targets_pred_std = surrogate_model.predict(train_x, num_samples=cfg.surrogate.ensemble_size)
            _, val_targets_pred_mean, val_targets_pred_std = surrogate_model.predict(val_x, num_samples=cfg.surrogate.ensemble_size)
            _, test_targets_pred_mean, test_targets_pred_std = surrogate_model.predict(test_x, num_samples=cfg.surrogate.ensemble_size)
        
        train_pred_dist = Normal(train_targets_pred_mean, train_targets_pred_std)
        train_nlpd = -1.0 * train_pred_dist.log_prob(
            torch.from_numpy(train_targets).to(train_targets_pred_mean)
        ).mean()

        val_pred_dist = Normal(val_targets_pred_mean, val_targets_pred_std)
        val_nlpd = -1.0 * val_pred_dist.log_prob(
            torch.from_numpy(val_targets).to(val_targets_pred_mean)
        ).mean()

        test_pred_dist = Normal(test_targets_pred_mean, test_targets_pred_std)
        test_nlpd = -1.0 * test_pred_dist.log_prob(
            torch.from_numpy(test_targets).to(test_targets_pred_mean)
        ).mean()
        
        train_targets_pred_mean = train_targets_pred_mean.cpu().numpy()
        test_targets_pred_mean = test_targets_pred_mean.cpu().numpy()
    else:
        raise NotImplementedError

    train_y_pred_mean = y_scaler.inverse_transform(train_targets_pred_mean)
    test_y_pred_mean = y_scaler.inverse_transform(test_targets_pred_mean)

    metrics = {
        "train_nlpd": train_nlpd,
        "val_nlpd": val_nlpd,
        "test_nlpd": test_nlpd,
        "train_target_rmse": np.sqrt(mean_squared_error(train_targets, train_targets_pred_mean)),
        "train_y_rmse": np.sqrt(mean_squared_error(train_y, train_y_pred_mean)),
        "train_y_mae": mean_absolute_error(train_y, train_y_pred_mean),
        "test_target_rmse": np.sqrt(mean_squared_error(test_targets, test_targets_pred_mean)),
        "test_y_rmse": np.sqrt(mean_squared_error(test_y, test_y_pred_mean)),
        "test_y_r2": r2_score(test_y, test_y_pred_mean),
        "test_y_mae": mean_absolute_error(test_y, test_y_pred_mean),
    }
    ret_val = metrics["test_nlpd"]
    metrics = {'/'.join([cfg.task.name, key]): val for key, val in metrics.items()}
    wandb.log(metrics)
    
    return ret_val


if __name__ == "__main__":
    main()
