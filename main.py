"""
main.py  –  CS 234 Negotiating Agents Project
──────────────────────────────────────────────
Entry point.  Run this file to train the MADDPG baseline.

Project structure
─────────────────
  negotiation_env.py   – PettingZoo Parallel environment (Rubinstein-inspired)
  maddpg.py            – MADDPG agent (Actor + centralised Critic)
  replay_buffer.py     – Joint-transition replay buffer
  train_baseline.py    – Training loop, evaluation, and plots

Usage
─────
  python main.py            # train baseline (5 000 episodes)
  python main.py --eval     # load saved checkpoints and run evaluation only
"""

import sys
from train_baseline import main

if __name__ == "__main__":
    eval_only = "--eval" in sys.argv
    main(eval_only=eval_only)
