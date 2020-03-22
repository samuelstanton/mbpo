params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'Hopper',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'baseline_svgp_v8',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,

        'real_ratio': 0.05,
        'target_entropy': -1,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 15],
        'rand_lengths': False,

        'model_type': 'GPEnsemble',
        'num_components': 7,
        'num_elites': 5,
        'hidden_dim': 512,
        'hidden_depth': 4,
        'n_inducing': 256,
        'max_epochs_since_update': 4
    }
}

