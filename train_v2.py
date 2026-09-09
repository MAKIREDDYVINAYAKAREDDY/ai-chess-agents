from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from chess_env_v4 import ChessEnvV4


print("=" * 70)
print("AI CHESS AGENT - V2.5 PPO TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# Create environment
# ------------------------------------------------------------

env = ChessEnvV4(
    max_moves=200
)

print("\nChecking environment...")
check_env(env, warn=True)
print("Environment check: PASSED")

print("\nObservation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="chess_ppo_v2"
)

# ------------------------------------------------------------
# PPO model
# ------------------------------------------------------------

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=0.0001,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    vf_coef=0.5,

    max_grad_norm=0.5,

    verbose=1,

    tensorboard_log="./tensorboard/"
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

TOTAL_TIMESTEPS = 100_000

print("\nStarting PPO training...")
print("Total timesteps:", TOTAL_TIMESTEPS)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True
)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

model.save(
    "chess_ai_agent_v2"
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nNew model:")
print("chess_ai_agent_v2.zip")

print("\nOriginal V1 model remains:")
print("chess_ai_agent.h5")

env.close()
