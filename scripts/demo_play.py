"""
Demo: watch agents play poker in the terminal with Unicode cards.
Runs N hands with rendering enabled so you can see the action.
Supports --search flag for MCTS-enhanced decisions.
"""
import argparse
import time
from pokerStats.rl import PokerEnv, Action, NUM_ACTIONS
from pokerStats.rl.ppo_agent import PPOAgent
from pokerStats.rl.self_play_trainer import ScriptedAgent
from pokerStats.rl.poker_env import hand_strength, draw_potential, raise_frac_to_amount
from pokerStats.rl.renderer import BOLD, RESET, GREEN, YELLOW, RED, GRAY, BLUE


ACTION_NAMES = {0: "FOLD", 1: "CHECK", 2: "CALL", 3: "RAISE", 4: "ALL-IN"}


def strength_bar(hs: float) -> str:
    """Visual hand strength bar."""
    n = int(hs * 10)
    if hs >= 0.7:
        col = GREEN
    elif hs >= 0.4:
        col = YELLOW
    else:
        col = RED
    bar = col + "\u2588" * n + GRAY + "\u2591" * (10 - n) + RESET
    return f"{bar} {hs:.0%}"


def demo(n_hands=10, delay=0.8, checkpoint=None, opponent_style="fish",
         num_players=3, use_search=False, num_simulations=100):
    env = PokerEnv(
        num_players=num_players,
        starting_stack=200.0,
        big_blind=2.0,
        render_mode="unicode",
    )

    use_alpha = checkpoint is not None or use_search
    agent = PPOAgent(use_alpha=use_alpha,
                     num_opponents=num_players - 1)
    if checkpoint:
        agent.load(checkpoint)
        agent.net.eval()
        label = f"Trained agent (ELO={agent.elo:.0f})"
    else:
        label = "Random agent (untrained)"

    # MCTS search
    mcts = None
    if use_search:
        from pokerStats.rl.mcts import PokerMCTS
        mcts = PokerMCTS(agent._cpu_net if not use_alpha else agent._cpu_net,
                         num_simulations=num_simulations)
        label += f" + MCTS ({num_simulations} sims)"

    opp = ScriptedAgent(opponent_style)
    sep = "=" * 58
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  Opponents: {num_players - 1}x {opponent_style}")
    print(f"  Actions: FOLD / CHECK / CALL / RAISE (continuous) / ALL-IN")
    print(f"  Hands: {n_hands}")
    print(f"  {sep}\n")

    wins = 0
    total_reward = 0.0

    for h in range(n_hands):
        obs = env.reset()
        done = False
        hand_actions = []

        while not done:
            pidx = env.current_player
            o = obs[pidx]
            mask = env.legal_mask()

            if pidx == 0:
                if mcts is not None:
                    # Use MCTS for hero decisions
                    action = mcts.get_action(env, 0, o, mask, temperature=0.1)
                    raise_frac = 0.3  # default for MCTS
                else:
                    action, raise_frac, _, _, _ = agent.get_action(o, mask)

                # Log hero decision with context
                hs = o[143]
                dp = o[144]
                act_name = ACTION_NAMES.get(action, "?")

                info_parts = [f"  {BOLD}Hero decides:{RESET} {act_name}"]
                if action == Action.RAISE:
                    call_amt = env._call_amount(0)
                    p = env.players[0]
                    min_r = max(env.last_bet * 2, call_amt + env.big_blind)
                    max_r = p.stack + p.current_bet
                    actual = raise_frac_to_amount(raise_frac, min_r, max_r)
                    info_parts.append(f"${actual:.0f} (frac={raise_frac:.2f})")

                info_parts.append(f"  strength: {strength_bar(hs)}")
                if dp > 0:
                    info_parts.append(f"  draw: {dp:.0%}")

                print("  " + "  ".join(info_parts))
            else:
                action, raise_frac, _, _, _ = opp.get_action(o, mask)

            obs, rewards, done, info = env.step(action, raise_frac)
            time.sleep(delay)

        hero_reward = env._rewards.get(0, 0.0)
        total_reward += hero_reward
        if hero_reward > 0:
            wins += 1

        if hero_reward > 0:
            result_str = f"{GREEN}WIN{RESET}"
        else:
            result_str = f"{RED}LOSS{RESET}"

        avg = total_reward / (h + 1)
        print(f"\n  Hand {h+1} result: {result_str}  ({hero_reward:+.1f} BB)")
        print(f"  Running: {wins}/{h+1} wins ({wins/(h+1)*100:.0f}%)  avg={avg:+.2f} BB/hand\n")
        time.sleep(delay * 2)

    print(f"\n  {sep}")
    print(f"  Final: {wins}/{n_hands} wins ({wins/n_hands*100:.0f}%)")
    print(f"  Total P&L: {total_reward:+.1f} BB")
    print(f"  {sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch poker agents play in the terminal")
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between actions")
    parser.add_argument("--checkpoint", default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--opponent", default="fish",
                        choices=["nit", "fish", "maniac", "tight-aggressive"],
                        help="Opponent style")
    parser.add_argument("--players", type=int, default=3, help="Number of players (2-6)")
    parser.add_argument("--search", action="store_true", help="Enable MCTS search")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per decision")
    args = parser.parse_args()

    demo(
        n_hands=args.hands,
        delay=args.delay,
        checkpoint=args.checkpoint,
        opponent_style=args.opponent,
        num_players=args.players,
        use_search=args.search,
        num_simulations=args.simulations,
    )
