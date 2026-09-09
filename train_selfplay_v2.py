import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from chess_env_v4 import ChessEnvV4


BASE_MODEL = "chess_ai_agent_v2.zip"
OUTPUT_MODEL = "chess_ai_agent_v2_selfplay"

TOTAL_TIMESTEPS = 100_000


def make_env():
    return ChessEnvV4(max_moves=200)


print("=" * 70)
print("AI CHESS AGENT - V2.7 ACTUAL SELF-PLAY TRAINING")
print("=" * 70)

print("\nLoading V2 PPO model...")

model = PPO.load(
    BASE_MODEL
)

print("Loaded:", BASE_MODEL)

# ------------------------------------------------------------
# Create fresh environment
# ------------------------------------------------------------

env = DummyVecEnv([
    make_env
])

model.set_env(env)

print("\nEnvironment:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

os.makedirs(
    "./selfplay_checkpoints",
    exist_ok=True
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./selfplay_checkpoints/",
    name_prefix="chess_selfplay"
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\nStarting self-play PPO training...")
print("Total timesteps:", TOTAL_TIMESTEPS)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
    progress_bar=True
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

model.save(
    OUTPUT_MODEL
)

print("\n" + "=" * 70)
print("SELF-PLAY TRAINING COMPLETE")
print("=" * 70)

print("\nSaved model:")
print(OUTPUT_MODEL + ".zip")

print("\nOriginal V2 model preserved:")
print(BASE_MODEL)

print("\nSelf-play checkpoints:")
print("./selfplay_checkpoints/")

env.close()
