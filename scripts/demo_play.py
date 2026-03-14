"""
Demo: watch agents play poker in the terminal with Unicode cards.
Runs N hands with rendering enabled so you can see the action.
"""
import argparse
import time
from pokerStats.rl import PokerEnv, NUM_ACTIONS
from pokerStats.rl.ppo_agent import PPOAgent
from pokerStats.rl.self_play_trainer import ScriptedAgent
from pokerStats.rl.renderer import render_training_stats


def demo(n_hands=10, delay=0.8, checkpoint=None, opponent_style="fish", num_players=3):
    env = PokerEnv(
        num_players=num_players,
        starting_stack=200.0,
        big_blind=2.0,
        render_mode="unicode",
    )

    # Load trained agent or use random
    agent = PPOAgent()
    if checkpoint:
        agent.load(checkpoint)
        agent.net.eval()
        label = f"Trained agent (ELO={agent.elo:.0f})"
    else:
        label = "Random agent (untrained)"

    opp = ScriptedAgent(opponent_style)
    print(f"\n  {'=' * 58}")
    print(f"  {label}")
    print(f"  Opponents: {num_players - 1}x {opponent_style}")
    print(f"  Hands: {n_hands}")
    print(f"  {'=' * 58}\n")

    wins = 0
    total_reward = 0.0

    for h in range(n_hands):
        obs = env.reset()
        done = False

        while not done:
            pidx = env.current_player
            o = obs[pidx]
            mask = env.legal_mask()

            if pidx == 0:
                action, _, _, _ = agent.get_action(o, mask)
            else:
                action, _, _, _ = opp.get_action(o, mask)

            obs, rewards, done, info = env.step(action)
            time.sleep(delay)

        hero_reward = env._rewards.get(0, 0.0)
        total_reward += hero_reward
        if hero_reward > 0:
            wins += 1

        result = "WIN" if hero_reward > 0 else "LOSS"
        print(f"\n  Hand {h + 1} result: {result}  ({hero_reward:+.1f} BB)")
        print(f"  Running: {wins}/{h + 1} wins  avg={total_reward / (h + 1):+.2f} BB/hand\n")
        time.sleep(delay * 2)

    print(f"\n  {'=' * 58}")
    print(f"  Final: {wins}/{n_hands} wins ({wins / n_hands * 100:.0f}%)")
    print(f"  Total P&L: {total_reward:+.1f} BB")
    print(f"  {'=' * 58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch poker agents play in the terminal")
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between actions")
    parser.add_argument("--checkpoint", default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--opponent", default="fish",
                        choices=["nit", "fish", "maniac", "tight-aggressive"],
                        help="Opponent style")
    parser.add_argument("--players", type=int, default=3, help="Number of players (2-6)")
    args = parser.parse_args()

    demo(
        n_hands=args.hands,
        delay=args.delay,
        checkpoint=args.checkpoint,
        opponent_style=args.opponent,
        num_players=args.players,
    )
