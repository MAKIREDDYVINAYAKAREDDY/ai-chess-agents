import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env


from chess_env_masked import MaskedChessEnv


OUTPUT_MODEL = "chess_ai_agent_v2_masked"

TOTAL_TIMESTEPS = 100_000


def mask_fn(env):
    return env.action_masks()


def make_env():
    env = MaskedChessEnv(
        max_moves=200
    )

    env = ActionMasker(
        env,
        mask_fn
    )

    return env


print("=" * 70)
print("AI CHESS AGENT - V2.8 MASKABLE PPO")
print("=" * 70)

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

env = make_env()

print("\nChecking environment...")

# ActionMasker exposes the underlying environment.
check_env(
    env.env,
    warn=True
)

print("Environment check: PASSED")

print("\nObservation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

print(
    "\nInitial legal actions:",
    int(env.action_masks().sum())
)

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

os.makedirs(
    "./masked_checkpoints",
    exist_ok=True
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./masked_checkpoints/",
    name_prefix="chess_masked"
)

# ------------------------------------------------------------
# Maskable PPO
# ------------------------------------------------------------

model = MaskablePPO(
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

    tensorboard_log="./tensorboard_masked/"
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\nStarting Maskable PPO training...")
print(
    "Total timesteps:",
    TOTAL_TIMESTEPS
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

model.save(
    OUTPUT_MODEL
)

print("\n" + "=" * 70)
print("MASKABLE PPO TRAINING COMPLETE")
print("=" * 70)

print("\nSaved:")
print(
    OUTPUT_MODEL + ".zip"
)

print("\nCheckpoints:")
print(
    "./masked_checkpoints/"
)

print("\nThe original models were not modified.")

env.close()
