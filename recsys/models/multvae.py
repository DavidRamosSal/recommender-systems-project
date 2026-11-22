"""Multinomial variational autoencoder from the Part 2 MovieLens notebook."""

from copy import deepcopy
from math import ceil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .item_knn import interaction_matrix


def _device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _Network(nn.Module):
    def __init__(self, n_items, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.encoder = nn.Linear(n_items, hidden_dim)
        self.posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_items)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.encoder, self.posterior, self.decoder, self.logits):
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)

    def forward(self, profiles):
        inputs = self.dropout(F.normalize(profiles, dim=1))
        mu, logvar = self.posterior(torch.tanh(self.encoder(inputs))).chunk(2, dim=1)
        latent = mu
        if self.training:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.logits(torch.tanh(self.decoder(latent))), mu, logvar


class MultVAE:
    def __init__(self, hidden_dim=600, latent_dim=200, dropout=0.5, beta_cap=0.2,
                 lr=0.001, weight_decay=0.0, batch_size=256, feedback="positives",
                 epochs=100, seed=42):
        if feedback not in ("positives", "all_ratings"):
            raise ValueError("feedback must be positives or all_ratings.")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_cap = beta_cap
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.feedback = feedback
        self.epochs = epochs
        self.seed = seed

    def fit(self, history, *, user_ids=None, catalogue=None, validation=None):
        """Early-stop on validation NDCG during tuning, refit for fixed epochs otherwise."""
        self.users = pd.Index(np.sort(history.user_id.unique()) if user_ids is None else user_ids)
        self.items = pd.Index(np.sort(history.item_id.unique()) if catalogue is None else catalogue)
        profiles = interaction_matrix(history, self.users, self.items,
                                      positives_only=self.feedback == "positives")
        device = _device()
        self.profiles = torch.from_numpy(profiles).to(device)
        training = self.profiles[torch.from_numpy(np.flatnonzero(profiles.any(axis=1))).to(device)]
        if not len(training):
            raise ValueError("Mult-VAE needs at least one nonempty training profile.")

        torch.manual_seed(self.seed)
        self.network = _Network(len(self.items), self.hidden_dim, self.latent_dim,
                                self.dropout).to(device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        shuffler = torch.Generator().manual_seed(self.seed)
        anneal_steps = 20 * ceil(len(training) / self.batch_size)
        step = 0
        best_ndcg = -np.inf
        best_state = None
        self.best_epoch = self.epochs
        self.validation_ndcg = []
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            perm = torch.randperm(len(training), generator=shuffler).to(device)
            for batch in perm.split(self.batch_size):
                x = training[batch]
                logits, mu, logvar = self.network(x)
                neg_ll = -(F.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
                loss = neg_ll + self.beta_cap * min(1., step / anneal_steps) * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
            if validation is not None:
                ndcg = validation(self)
                self.validation_ndcg.append(float(ndcg))
                if ndcg > best_ndcg:
                    best_ndcg, self.best_epoch = ndcg, epoch
                    best_state = deepcopy(self.network.state_dict())
                elif epoch - self.best_epoch >= 10:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score_users(self, user_ids, item_ids=None):
        rows = self.users.get_indexer(user_ids)
        if (rows < 0).any():
            raise ValueError("Unknown user ID. Include all evaluation users in user_ids at fit.")
        self.network.eval()
        rows = torch.from_numpy(rows).to(self.profiles.device)
        scores = torch.cat([self.network(self.profiles[batch])[0] for batch in rows.split(1024)])
        scores = scores.cpu().numpy()
        if item_ids is None:
            return scores
        columns = self.items.get_indexer(item_ids)
        result = np.full((len(rows), len(columns)), -np.inf, dtype=scores.dtype)
        known = columns >= 0
        result[:, known] = scores[:, columns[known]]
        return result

    def score(self, user_id, item_ids):
        return self.score_users([user_id], item_ids)[0].astype(np.float64)

    def weights(self):
        return {"users": self.users.to_numpy(), "items": self.items.to_numpy(),
                "profiles": self.profiles.cpu().numpy().astype(bool),
                **{name: tensor.cpu().numpy() for name, tensor in self.network.state_dict().items()}}

    def load_weights(self, weights):
        self.users, self.items = pd.Index(weights["users"]), pd.Index(weights["items"])
        device = _device()
        self.profiles = torch.from_numpy(weights["profiles"]).to(device, dtype=torch.float32)
        self.network = _Network(len(self.items), self.hidden_dim, self.latent_dim,
                                self.dropout).to(device)
        self.network.load_state_dict({k: torch.from_numpy(weights[k])
                                      for k in self.network.state_dict()})
        self.network.eval()


def validation_configs(budget, seed=42):
    """Notebook search order: reference model first, then distinct seeded draws."""
    grid = {
        "hidden_dim": [200, 400, 600, 800], "latent_dim": [32, 64, 128, 200],
        "dropout": [0.2, 0.3, 0.5, 0.7], "beta_cap": [0.05, 0.1, 0.2, 0.4, 1.0],
        "lr": [0.0003, 0.0005, 0.001, 0.002, 0.003], "weight_decay": [0., 1e-6, 1e-5, 1e-4],
        "batch_size": [128, 256, 512], "feedback": ["positives", "all_ratings"],
    }
    reference = MultVAE()
    configs = [{key: getattr(reference, key) for key in grid}]
    rng = np.random.default_rng(seed)
    while len(configs) < budget:
        config = {key: values[rng.integers(len(values))] for key, values in grid.items()}
        if config not in configs:
            configs.append(config)
    return configs
