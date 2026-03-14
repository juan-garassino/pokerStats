"""Tests for pokerStats.rl.mcts"""
import numpy as np
import torch
from pokerStats.rl.poker_env import PokerEnv, Action, NUM_ACTIONS, OBS_DIM
from pokerStats.rl.ppo_agent import AlphaPokerNet
from pokerStats.rl.mcts import PokerMCTS, MCTSNode


def test_mcts_node_ucb():
    """MCTSNode computes UCB score."""
    root = MCTSNode()
    root.visit_count = 10
    child = MCTSNode(parent=root, action=0, prior=0.5)
    child.visit_count = 3
    child.value_sum = 1.5

    score = child.ucb_score(c_puct=1.5)
    assert score > 0  # should have both exploitation and exploration components


def test_mcts_node_expand():
    """MCTSNode expansion creates children."""
    node = MCTSNode()
    priors = {0: 0.3, 1: 0.5, 2: 0.2}
    node.expand(priors)
    assert len(node.children) == 3
    assert node.children[1].prior == 0.5


def test_mcts_node_backup():
    """Backup propagates value up the tree."""
    root = MCTSNode()
    root.expand({0: 0.5, 1: 0.5})
    child = root.children[0]
    child.backup(1.0)

    assert child.visit_count == 1
    assert child.value_sum == 1.0
    assert root.visit_count == 1
    assert root.value_sum == -1.0  # flipped


def test_mcts_search_produces_valid_actions():
    """MCTS search returns a valid action distribution."""
    net = AlphaPokerNet(num_opponents=2)
    net.eval()
    mcts = PokerMCTS(net, num_simulations=10, c_puct=1.5)

    env = PokerEnv(num_players=3)
    env.reset()

    obs = env._build_obs(env.current_player)
    mask = env.legal_mask()

    action_probs = mcts.search(env, env.current_player, obs, mask)

    assert action_probs.shape == (NUM_ACTIONS,)
    assert action_probs.sum() > 0.99  # should sum to ~1
    # Only legal actions should have probability
    for i in range(NUM_ACTIONS):
        if not mask[i]:
            assert action_probs[i] == 0.0


def test_mcts_get_action():
    """MCTS get_action returns a valid legal action."""
    net = AlphaPokerNet(num_opponents=2)
    net.eval()
    mcts = PokerMCTS(net, num_simulations=10)

    env = PokerEnv(num_players=3)
    env.reset()

    obs = env._build_obs(env.current_player)
    mask = env.legal_mask()
    legal = env.legal_actions()

    action = mcts.get_action(env, env.current_player, obs, mask, temperature=0.0)
    assert action in legal


def test_mcts_does_not_modify_env():
    """MCTS search should not modify the original env state."""
    net = AlphaPokerNet(num_opponents=2)
    net.eval()
    mcts = PokerMCTS(net, num_simulations=5)

    env = PokerEnv(num_players=3)
    env.reset()

    pot_before = env.pot
    player_before = env.current_player
    community_before = list(env.community)

    obs = env._build_obs(env.current_player)
    mask = env.legal_mask()
    mcts.search(env, env.current_player, obs, mask)

    assert env.pot == pot_before
    assert env.current_player == player_before
    assert env.community == community_before
