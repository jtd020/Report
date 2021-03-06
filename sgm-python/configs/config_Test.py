dict(
    seed=0,
    env=dict(
        env_name='test',
        max_episode_steps=20,
        resize_factor=2, # Inflate the environment to increase the difficulty.
        thin=False, # If True, resize by expanding open space, not walls, to make walls thin
    ),
    runner=dict(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps=2,
        opt_steps=1,
        batch_size_opt=64,

        num_eval_episodes=10,
        eval_interval=1000,
    ),
    agent=dict(
        discount=1,
        num_bins=20, # equal to max_episode_steps
        use_distributional_rl=True,
        ensemble_size=3,
        targets_update_interval=5, # tfagents default
        tau=0.05,
    ),
    replay_buffer=dict(
        max_size=1000,
    ),
    ckpt_dir='./workdirs/testrand/',
)