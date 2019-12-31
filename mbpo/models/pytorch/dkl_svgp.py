import copy
import math
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.models import GP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from mbpo.models.pytorch.fc import PytorchFC


class DeepFeatureSVGP(GP):
    def __init__(
            self,
            input_size: int,
            feature_size: int,
            label_size: int,
            hidden_width: int or list,
            hidden_depth: int,
            n_inducing: int,
            batch_size: int,
            max_epochs_since_update,
            **kwargs
    ):
        params = locals()
        del params['self']
        self.__dict__ = params
        super().__init__()

        noise_constraint = GreaterThan(1e-4)
        self.likelihood = GaussianLikelihood(
            batch_shape=torch.Size([label_size]),
            noise_constraint=noise_constraint
        )

        self.nn = PytorchFC(
            input_shape=torch.Size([input_size]),
            output_shape=torch.Size([feature_size]),
            hidden_width=hidden_width,
            depth=hidden_depth,
            batch_norm=False
        )
        self.nn_feature_mean = torch.zeros(feature_size)
        self.nn_feature_std = torch.ones(feature_size)
        self.mean_module = ConstantMean(batch_shape=torch.Size([label_size]))
        base_kernel = RBFKernel(
            batch_shape=torch.Size([label_size]),
            ard_num_dims=feature_size
        )
        self.covar_module = ScaleKernel(base_kernel, batch_shape=torch.Size([label_size]))
        self.batch_norm = torch.nn.BatchNorm1d(feature_size)

        variational_dist = MeanFieldVariationalDistribution(
            n_inducing, torch.Size([label_size])
        )
        inducing_points = torch.randn(n_inducing, input_size)
        self.variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )

    def _train(self):
        return GP.train(self, True)

    def eval(self):
        return GP.train(self, False)

    def forward(self, inputs):
        features = self.nn(inputs)
        features = self.batch_norm(features)
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        return MultivariateNormal(mean, covar)

    def __call__(self, inputs):
        return self.variational_strategy(inputs)

    def predict(
            self,
            inputs: np.ndarray,
            latent=False
    ):
        inputs = torch.tensor(inputs, dtype=torch.get_default_dtype())
        with torch.no_grad():
            pred_dist = self(inputs)
        if not latent:
            pred_dist = self.likelihood(pred_dist)
        pred_mean = pred_dist.mean.t().cpu().numpy()
        pred_var = pred_dist.variance.t().cpu().numpy()
        return pred_mean, pred_var

    def train(
            self,
            inputs: np.ndarray,
            labels: np.ndarray,
            pretrain=False,
            holdout_ratio=0.,
            max_epochs=None,
            max_kl=None,
            objective='elbo',
            early_stopping=False,
            **kwargs
    ):
        metrics = {
            'train_loss': [],
            'holdout_loss': [],
        }
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(inputs, dtype=torch.get_default_dtype()),
            torch.tensor(labels, dtype=torch.get_default_dtype())
        )
        if pretrain:
            if self.feature_size == self.label_size:
                print("pretraining feature extractor")
                self.nn.fit(dataset)
            else:
                raise RuntimeError("features and labels must be the same size to pretrain")

        if early_stopping and holdout_ratio <= 0.:
            raise ValueError("holdout dataset required for early stopping")

        n_val = min(int(2048), int(holdout_ratio * len(dataset)))
        if n_val > 0:
            n_train = len(dataset) - n_val
            train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
            val_x, val_y = val_data[:]
        else:
            train_data, val_data = dataset, None

        if max_kl:
            assert holdout_ratio > 0
            self.eval()
            with torch.no_grad():
                ref_pred = self(val_x)
                ref_pred = torch.distributions.Normal(ref_pred.mean, ref_pred.variance.sqrt())

        dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        if objective == 'elbo':
            obj = VariationalELBO(self.likelihood, self, num_data=len(dataset))
        elif objective == 'pll':
            obj = PredictiveLogLikelihood(self.likelihood, self, num_data=len(dataset), beta=1e-3)
        optimizer = Adam(self.optim_param_groups)

        exit_training = False
        num_batches = math.ceil(len(dataset) / self.batch_size)
        epoch = 1
        snapshot = (1, 1e6, self.state_dict())
        avg_train_loss = None
        alpha = 2 / (num_batches + 1)

        print(f"training w/ objective {objective} on {len(train_data)} examples")
        while not exit_training:
            self._train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                out = self(inputs)
                loss = -obj(out, labels.t()).sum()
                loss.backward()
                optimizer.step()

                if avg_train_loss:
                    avg_train_loss = alpha * loss.detach() + (1 - alpha) * avg_train_loss
                else:
                    avg_train_loss = loss.detach()

            conv_metric = avg_train_loss
            if val_data:
                with torch.no_grad():
                    self.eval()
                    holdout_pred = self(val_x)
                    holdout_loss = -obj(holdout_pred, val_y.t()).sum()
                    metrics['holdout_loss'].append(holdout_loss)
                if max_kl:
                    holdout_pred = torch.distributions.Normal(holdout_pred.mean, holdout_pred.variance.sqrt())
                    holdout_kl = torch.distributions.kl.kl_divergence(holdout_pred, ref_pred).mean()

            if early_stopping:
                conv_metric = holdout_loss
            snapshot, exit_training = self.save_best(snapshot, epoch, conv_metric)
            epoch += 1
            metrics['train_loss'].append(avg_train_loss)
            if exit_training or (max_epochs and epoch == max_epochs):
                print(f"Training converged after {epoch} epochs, halting")
                break
            if max_kl and holdout_kl > max_kl:
                print(f"max kl threshold exceeded, exiting training")
                break
        self.load_state_dict(snapshot[2])
        self.eval()
        return metrics

    def save_best(self, snapshot, epoch, avg_train_loss):
        exit_training = False
        last_update, best_loss, _ = snapshot
        improvement = (best_loss - avg_train_loss) / max(abs(best_loss), 1.)
        if improvement > 0.01:
            snapshot = (epoch, avg_train_loss.item(), self.state_dict())
        if epoch == snapshot[0] + self.max_epochs_since_update:
            exit_training = True
        return snapshot, exit_training

    def set_inducing_loc(self, inputs):
        idx = torch.randint(0, inputs.shape[0], (self.n_inducing,))
        self.variational_strategy.inducing_points.data.copy_(inputs[idx])

    def make_copy(self):
        new_model = DeepFeatureSVGP(**self.__dict__)
        new_model.load_state_dict(self.state_dict())
        return new_model

    @property
    def optim_param_groups(self):
        gp_params, nn_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if 'nn' in name:
                nn_params.append(param)
            elif 'raw' in name:
                gp_params.append(param)
            else:
                other_params.append(param)

        gp_params = {
            'params': gp_params,
            'lr': 1e-2
        }
        other_params = {
            'params': other_params,
            'lr': 1e-3
        }
        nn_params = {
            'params': nn_params,
            'lr': 1e-4
        }
        return [gp_params, other_params, nn_params]
