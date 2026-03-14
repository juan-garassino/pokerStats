from os.path import isfile
from os.path import dirname

version_file = "{}/version.txt".format(dirname(__file__))

if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()

# RL subpackage convenience imports
from pokerStats.rl import (
    PokerEnv, Action, NUM_ACTIONS, OBS_DIM,
    PPOAgent, RolloutBuffer, PokerNet,
)
