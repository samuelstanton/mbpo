import numpy as np
import tensorflow as tf
import torch

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN
from mbpo.models.pytorch.dkl_svgp import DeepFeatureSVGP
from mbpo.models.pytorch.bnn import PytorchBNN


def construct_model(
		model_type='TensorflowBNN',
		obs_dim=11,
		act_dim=3,
		rew_dim=1,
		hidden_dim=200,
		num_networks=7,
		num_elites=5,
		n_inducing=256,
		session=None
):
	print(f"[ {model_type} ] Observation dim {obs_dim} | Action dim: {act_dim} | Hidden dim: {hidden_dim}")
	if model_type == 'TensorflowBNN':
		params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
		model = BNN(params)
		model.add(FC(hidden_dim, input_dim=obs_dim+act_dim, activation="swish", weight_decay=0.000025))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))
		model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

	elif model_type == 'PytorchBNN':
		model = PytorchBNN(
            input_shape=torch.Size([obs_dim + act_dim]),
            label_shape=torch.Size([rew_dim + obs_dim]),
            hidden_width=hidden_dim,
            hidden_depth=4,
            ensemble_size=num_networks,
            minibatch_size=256,
            lr=1e-3,
            logvar_penalty_coeff=1e-2,
            max_epochs_since_update=5,
        )

	elif model_type == 'DeepFeatureSVGP':
		model = DeepFeatureSVGP(
			input_dim=obs_dim + act_dim,
			feature_dim=rew_dim + obs_dim,
			label_dim=rew_dim + obs_dim,
			hidden_width=hidden_dim,
			hidden_depth=2,
			n_inducing=n_inducing,
			batch_size=256,
			max_epochs_since_update=4
		)

	else:
		raise RuntimeError("unrecognized model type")

	return model


def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs


def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))


if __name__ == '__main__':
	model = construct_model()
