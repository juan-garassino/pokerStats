"""
Quick RL evaluation script.
Trains a short session, then benchmarks against scripted opponents.
"""
import numpy as np
from collections import deque
from pokerStats.rl import PokerEnv, NUM_ACTIONS
from pokerStats.rl.ppo_agent import PPOAgent, RolloutBuffer
from pokerStats.rl.self_play_trainer import ScriptedAgent


# ── Benchmark: play N hands against a specific opponent style ──────────────
def benchmark(agent, style, n_hands=2000, num_players=6):
    env = PokerEnv(num_players=num_players, starting_stack=200.0, big_blind=2.0)
    rewards = []
    wins = 0

    for _ in range(n_hands):
        obs = env.reset()
        done = False
        opp = ScriptedAgent(style)

        while not done:
            pidx = env.current_player
            o = obs[pidx]
            mask = env.legal_mask()
            if pidx == 0:
                action, _, _, _ = agent.get_action(o, mask, deterministic=True)
            else:
                action, _, _, _ = opp.get_action(o, mask)
            obs, rews, done, _ = env.step(action)

        hero_reward = env._rewards.get(0, 0.0)
        rewards.append(hero_reward)
        if hero_reward > 0:
            wins += 1

    bb = 2.0
    mean_r = np.mean(rewards)
    bb_per_100 = (mean_r / bb) * 100
    win_pct = wins / n_hands * 100
    return {"style": style, "mean_reward": mean_r, "bb/100": bb_per_100,
            "win_pct": win_pct, "n_hands": n_hands}


# ── Quick train ──────────────────────────────────────────────────────────
def quick_train(n_hands=5000, num_players=6):
    print(f"Training PPO agent for {n_hands} hands (self-play, {num_players} players)...")
    env = PokerEnv(num_players=num_players, starting_stack=200.0, big_blind=2.0)
    agent = PPOAgent(lr=2.5e-4)
    buf = RolloutBuffer(2048)
    reward_hist = deque(maxlen=500)

    for h in range(n_hands):
        obs = env.reset()
        done = False
        while not done:
            pidx = env.current_player
            o = obs[pidx]
            mask = env.legal_mask()
            action, log_prob, value, entropy = agent.get_action(o, mask)
            obs_next, rewards, done, _ = env.step(action)

            if pidx == 0:
                reward = rewards.get(0, 0.0)
                buf.add(o, action, log_prob, reward, value, float(done), mask)

                if buf.full:
                    agent.update(buf, ppo_epochs=4, batch_size=256)

            obs = obs_next

        hero_reward = env._rewards.get(0, 0.0)
        reward_hist.append(hero_reward)

        if (h + 1) % 500 == 0:
            mr = np.mean(reward_hist)
            wr = sum(1 for r in reward_hist if r > 0) / len(reward_hist) * 100
            print(f"  Hand {h+1:5d} | mean_reward: {mr:+.2f} | win%: {wr:.1f}%")

    return agent


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hands", type=int, default=5000)
    parser.add_argument("--eval-hands", type=int, default=2000)
    parser.add_argument("--skip-train", action="store_true", help="Benchmark a random agent (no training)")
    args = parser.parse_args()

    if args.skip_train:
        print("Benchmarking UNTRAINED (random) agent...")
        agent = PPOAgent()
    else:
        agent = quick_train(n_hands=args.train_hands)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.eval_hands} hands per opponent style")
    print(f"{'='*60}")

    styles = ["nit", "fish", "maniac", "tight-aggressive"]
    for style in styles:
        result = benchmark(agent, style, n_hands=args.eval_hands)
        print(f"  vs {style:20s} | bb/100: {result['bb/100']:+7.1f} | win%: {result['win_pct']:.1f}%")

    print(f"{'='*60}")
