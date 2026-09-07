"""Focused invariants of the frozen-model prior experiment."""

import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn

from pipelines.fit_prior import _finite, pair_scores
from src.distributions import Gaussian
from src.training.prior_fit import (
    LatentCache,
    PriorFitConfig,
    fit_prior,
    matching_loss,
    recorded_pairs,
    select_pairs,
)


class TinyModel(nn.Module):
    """A Gaussian prior beside independently frozen model components."""

    def __init__(self):
        super().__init__()
        self.prior = TinyPrior()
        self.encoder = nn.Linear(in_features=2, out_features=2)
        self.posterior = nn.Linear(in_features=2, out_features=2)
        self.decoder = nn.Linear(in_features=2, out_features=2)
        self.register_buffer('constant', torch.ones(2))


class TinyPrior(nn.Module):
    """A Gaussian head with dropout to expose accidental training-mode activation."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.9)
        self.head = nn.Linear(in_features=2, out_features=4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, summary):
        return Gaussian.from_head(self.head(self.dropout(summary)))


class PairDataset:
    """Minimal indexed case identities for subset-selection tests."""

    def __len__(self):
        return 8

    def _get_cut(self, index):
        return SimpleNamespace(case_id=str(index // 2)), index % 2 + 1


class PriorFitTests(unittest.TestCase):
    """Verify fitting, deterministic pairing, and exclusion of validation cases."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = PriorFitConfig.model_validate(
            yaml.safe_load(Path('config/experiments/prior-only-fit.yaml').read_text())
        ).model_copy(update={'steps': 1, 'batch_size': 4, 'measurement_steps': (0, 1)})
        self.cache = LatentCache(
            summaries=torch.ones(size=(4, 2)),
            posterior=Gaussian(
                mean=torch.full(size=(4, 2), fill_value=0.2), logvar=torch.zeros(size=(4, 2))
            ),
        )

    def test_exact_kl_below_free_bits_and_frozen_update(self):
        model = TinyModel().eval()
        before = deepcopy(model.state_dict())
        loss = matching_loss(model=model, cache=self.cache)
        self.assertAlmostEqual(loss.item(), 0.04, places=6)
        q = torch.distributions.Normal(
            loc=self.cache.posterior.mean, scale=(self.cache.posterior.logvar / 2).exp()
        )
        p = torch.distributions.Normal(loc=torch.zeros(size=(4, 2)), scale=torch.ones(size=(4, 2)))
        expected = torch.distributions.kl_divergence(q, p).sum(dim=-1).mean()
        torch.testing.assert_close(loss, expected)
        history = fit_prior(
            model=model,
            cache=self.cache,
            config=self.config,
            measurement_caches={'train': self.cache, 'matching_validation': self.cache},
        )
        self.assertEqual(len(history['updates']), 1)
        self.assertEqual([row['step'] for row in history['measurements']], [0, 1])
        self.assertEqual(
            set(history['measurements'][0]['kl_nats']), {'train', 'matching_validation'}
        )
        self.assertLess(matching_loss(model=model, cache=self.cache).item(), loss.item())
        self.assertFalse(any(module.training for module in model.modules()))
        for name, value in model.state_dict().items():
            if not name.startswith('prior.'):
                self.assertTrue(torch.equal(value, before[name]), name)
        for name, parameter in model.named_parameters():
            if not name.startswith('prior.'):
                self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(model.prior.head.weight, before['prior.head.weight']))
        repeat = TinyModel()
        repeat.load_state_dict(before)
        fit_prior(
            model=repeat,
            cache=self.cache,
            config=self.config,
            measurement_caches={'train': self.cache, 'matching_validation': self.cache},
        )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, repeat.state_dict()[name]))

    def test_subset_exclusion_and_repeatability(self):
        dataset = PairDataset()
        first = select_pairs(dataset, count=4, seed=42, excluded_cases={'1'})
        second = select_pairs(dataset, count=4, seed=42, excluded_cases={'1'})
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(set(first.indices)), 4)
        self.assertTrue(all(dataset._get_cut(i)[0].case_id != '1' for i in first.indices))
        with self.assertRaises(ValueError):
            select_pairs(dataset, count=7, seed=42, excluded_cases={'1'})

    def test_recorded_subset_rejects_stale_identities(self):
        dataset = PairDataset()
        subset = recorded_pairs(
            dataset=dataset,
            identities=[
                {'index': 0, 'case_id': '0', 'prefix_len': 1},
                {'index': 3, 'case_id': '1', 'prefix_len': 2},
            ],
        )
        self.assertEqual(subset.indices, [0, 3])
        with self.assertRaises(ValueError):
            recorded_pairs(
                dataset=dataset,
                identities=[{'index': 0, 'case_id': 'wrong', 'prefix_len': 1}],
            )

    def test_pairing_and_unavailable_scores(self):
        row = {'case_id': 'a', 'prefix_len': 2, 'suffix_len': 3, 'comparable': True, 'emsc': 0.5}
        paired = pair_scores(original=[row], fitted=[row | {'emsc': 0.6}])
        self.assertEqual(paired[0]['original/emsc'], 0.5)
        self.assertEqual(paired[0]['fitted/emsc'], 0.6)
        with self.assertRaises(ValueError):
            pair_scores(original=[row], fitted=[row | {'case_id': 'b'}])
        with self.assertRaises(ValueError):
            pair_scores(original=[row], fitted=[])
        self.assertEqual(
            _finite({'scores': [float('nan'), float('inf')]}), {'scores': [None, None]}
        )


if __name__ == '__main__':
    unittest.main()
