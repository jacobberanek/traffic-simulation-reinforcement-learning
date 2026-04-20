"""
train.py — Train the Q-learning agent and compare against baselines.

Usage
-----
    python train.py                   # train + evaluate, print table
    python train.py --visual          # watch the trained agent run live

Outputs
-------
    qtable.pkl        — saved Q-table (reusable across sessions)
    results.csv       — per-episode stats for all three modes
"""

import argparse
import csv
import os

from QLearningAgent import QLearningAgent
from Simulator import Simulator


# ── Hyperparameters ───────────────────────────────────────────────────────────

TRAIN_EPISODES = 500    # how many episodes to train the agent
EVAL_EPISODES  = 20     # how many episodes to evaluate each mode
EPISODE_TICKS  = 300    # length of each episode in simulator ticks
EVAL_SEED      = 42     # fixed seed so all three modes see identical traffic
QTABLE_PATH    = "qtable.pkl"
RESULTS_PATH   = "results.csv"


# ── Training loop ─────────────────────────────────────────────────────────────

def train(agent: QLearningAgent) -> None:
    """Run TRAIN_EPISODES episodes and update the agent's Q-table."""
    print(f"\n{'='*60}")
    print(f"Training Q-learning agent for {TRAIN_EPISODES} episodes...")
    print(f"{'='*60}")

    sim = Simulator(duration=EPISODE_TICKS, delay=0, agent=agent, seed=None)

    for ep in range(1, TRAIN_EPISODES + 1):
        sim.reset()
        sim.run(visual=False, verbose=False)
        agent.end_episode()

        if ep % 25 == 0:
            agent.print_stats()

    print("\nTraining complete.")
    agent.print_stats()
    agent.save(QTABLE_PATH)

# ── Training loop (visuals for Episode 1 and Episode 200) ──────────────────────

def train_with_demo(agent: QLearningAgent) -> None:
    print("\n--- Episode 1: Untrained agent (visual) ---")
    sim = Simulator(duration=EPISODE_TICKS, delay=0.25, agent=agent, seed=EVAL_SEED)
    sim.run(visual=True)
    agent.end_episode()

    print(f"\n--- Episodes 2-{TRAIN_EPISODES - 1}: Headless training ---")
    sim_fast = Simulator(duration=EPISODE_TICKS, delay=0, agent=agent, seed=None)
    for ep in range(2, TRAIN_EPISODES):
        sim_fast.reset()
        sim_fast.run(visual=False, verbose=False)
        agent.end_episode()
        if ep % 25 == 0:
            agent.print_stats()

    print(f"\n--- Episodes {TRAIN_EPISODES}-{TRAIN_EPISODES + 1}: Trained agent (visual) ---")
    agent.epsilon = 0.0
    for ep in range(TRAIN_EPISODES, TRAIN_EPISODES + 2):
        sim_demo = Simulator(duration=EPISODE_TICKS, delay=0.25, agent=agent, seed=EVAL_SEED)
        sim_demo.run(visual=True)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_mode(mode: str, agent: QLearningAgent | None = None) -> list[dict]:
    """
    Run EVAL_EPISODES episodes for a given mode and return per-episode stats.

    mode : "qlearning" | "fixed" | "random"
    """
    results = []

    for ep in range(1, EVAL_EPISODES + 1):
        # Use a different seed each episode but keep it deterministic so
        # results are reproducible.  All three modes use the same seed
        # sequence so they face identical traffic.
        episode_seed = EVAL_SEED + ep

        if mode == "qlearning":
            # Greedy evaluation — no exploration
            saved_epsilon    = agent.epsilon
            agent.epsilon    = 0.0
            sim = Simulator(duration=EPISODE_TICKS, delay=0,
                            agent=agent, seed=episode_seed)
            stats = sim.run(visual=False, verbose=False)
            agent.epsilon = saved_epsilon  # restore so training can resume

        elif mode == "fixed":
            sim = Simulator(duration=EPISODE_TICKS, delay=0,
                            agent=None, fixed_switch_interval=10,
                            seed=episode_seed)
            stats = sim.run(visual=False, verbose=False)

        elif mode == "random":
            from QLearningAgent import QLearningAgent as _QL, ACTIONS
            import random as _r

            # A "random agent" just picks KEEP or SWITCH randomly every tick
            class RandomAgent:
                epsilon = 0.0  # duck-type compatibility
                def choose_action(self, state): return _r.Random(None).choice(ACTIONS)
                def update(self, *args): pass

            sim = Simulator(duration=EPISODE_TICKS, delay=0,
                            agent=RandomAgent(), seed=episode_seed)
            stats = sim.run(visual=False, verbose=False)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        stats["mode"]    = mode
        stats["episode"] = ep
        results.append(stats)

    return results


# ── Results table ─────────────────────────────────────────────────────────────

def summarise(all_results: list[dict]) -> None:
    """Print a comparison table and save results to CSV."""

    # Write CSV
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "episode",
            "avg_wait", "total_processed", "total_remaining",
        ])
        for r in all_results:
            writer.writerow([
                r["mode"], r["episode"],
                f"{r['avg_wait']:.2f}",
                r["total_processed"],
                r["total_remaining"],
            ])
    print(f"\nPer-episode results saved to {RESULTS_PATH}")

    # Console summary table
    print(f"\n{'='*60}")
    print(f"{'Mode':<12} {'Avg wait (ticks)':>18} {'Cars processed':>16} {'Remaining':>10}")
    print(f"{'-'*60}")

    for mode in ["fixed", "random", "qlearning"]:
        rows = [r for r in all_results if r["mode"] == mode]
        avg_wait  = sum(r["avg_wait"]        for r in rows) / len(rows)
        processed = sum(r["total_processed"] for r in rows) / len(rows)
        remaining = sum(r["total_remaining"] for r in rows) / len(rows)
        print(f"{mode:<12} {avg_wait:>18.2f} {processed:>16.1f} {remaining:>10.1f}")

    print(f"{'='*60}")
    print(f"(averages over {EVAL_EPISODES} evaluation episodes each)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual",     action="store_true",
                        help="Watch the trained agent run after evaluation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load existing qtable.pkl")
    args = parser.parse_args()

    # Load or train agent
    if args.skip_train and os.path.exists(QTABLE_PATH):
        agent = QLearningAgent.load(QTABLE_PATH)
    else:
        agent = QLearningAgent(
            alpha         = 0.1,
            gamma         = 0.95,
            epsilon       = 1.0,
            epsilon_min   = 0.05,
            epsilon_decay = 0.97,
        )
        train(agent) #train(agent) for testing, #train_with_demo(agent) for demo

    # Evaluate all three modes
    print(f"\nEvaluating all modes over {EVAL_EPISODES} episodes each...")
    all_results = []
    for mode in ["fixed", "random", "qlearning"]:
        print(f"  Running {mode}...")
        all_results.extend(evaluate_mode(mode, agent if mode == "qlearning" else None))

    summarise(all_results)

    # Optional: watch the trained agent live
    if args.visual:
        print("Launching visual demo of trained agent...")
        agent.epsilon = 0.0
        sim = Simulator(duration=EPISODE_TICKS, delay=0.15,
                        agent=agent, seed=EVAL_SEED)
        sim.run(visual=True)


if __name__ == "__main__":
    main()