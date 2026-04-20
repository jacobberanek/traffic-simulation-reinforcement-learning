import random
import pickle


# ── Action constants ──────────────────────────────────────────────────────────
KEEP   = "KEEP"    # Hold the current light phase for another tick
SWITCH = "SWITCH"  # Request a light phase change via trigger_light_switch()
ACTIONS = [KEEP, SWITCH]


def bucket(queue_length: int) -> int:
    """
    Discretise a raw queue length into one of four levels so the Q-table
    stays a manageable size.

        0 cars       -> 0  (empty)
        1-2 cars     -> 1  (light)
        3-5 cars     -> 2  (moderate)
        6+ cars      -> 3  (heavy)

    With 4 directions x 4 buckets x 4 light-states the full state space is
    4^4 x 4 = 1 024 states — trivial for a dictionary-based Q-table.
    """
    if queue_length == 0:
        return 0
    elif queue_length <= 2:
        return 1
    elif queue_length <= 5:
        return 2
    else:
        return 3


class QLearningAgent:
    """
    Tabular Q-learning agent for adaptive traffic signal control.

    The agent observes a discretised snapshot of the intersection each tick
    and chooses to either KEEP the current light phase or SWITCH it.  It
    updates a dictionary-based Q-table using the standard Bellman equation
    after every transition.

    Parameters
    ----------
    alpha : float
        Learning rate — how strongly each new experience overwrites the
        existing Q-value estimate.  Typical range: 0.05-0.3.
    gamma : float
        Discount factor — how much the agent values future rewards relative
        to immediate ones.  1.0 means fully far-sighted; 0.0 means myopic.
    epsilon : float
        Initial exploration probability for ε-greedy action selection.
    epsilon_min : float
        Floor below which epsilon will not decay.
    epsilon_decay : float
        Multiplicative decay applied to epsilon at the end of every episode.
    """

    def __init__(
        self,
        alpha:         float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.99,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps (state_tuple, action_str) → float
        # Missing entries default to 0.0 (optimistic initialisation is also
        # common, but zero works fine here).
        self.q_table: dict[tuple, float] = {}

        # ── Episode-level tracking ────────────────────────────────────────
        self.episode          = 0
        self.episode_rewards: list[float] = []   # sum of rewards per episode
        self._current_episode_reward = 0.0

    # ── Core RL interface ─────────────────────────────────────────────────────

    def get_q(self, state: tuple, action: str) -> float:
        """Return Q(state, action), defaulting to 0.0 if unseen."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: tuple) -> str:
        """
        ε-greedy action selection.

        With probability epsilon pick a random action (explore); otherwise
        pick the action with the highest Q-value for this state (exploit).
        Ties in the greedy case are broken randomly.
        """
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        # Greedy: pick action with highest Q-value
        q_values = {a: self.get_q(state, a) for a in ACTIONS}
        max_q    = max(q_values.values())
        # Collect all actions that tie for the max, then pick randomly among them
        best     = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best)

    def update(
        self,
        state:      tuple,
        action:     str,
        reward:     float,
        next_state: tuple,
    ) -> None:
        """
        Apply the Bellman update to Q(state, action):

            Q(s,a) ← Q(s,a) + α × [ r + γ·max_a' Q(s',a') − Q(s,a) ]

        Parameters
        ----------
        state      : the state in which the action was taken
        action     : the action that was taken (KEEP or SWITCH)
        reward     : the immediate reward received
        next_state : the state observed after the action
        """
        current_q  = self.get_q(state, action)
        best_next_q = max(self.get_q(next_state, a) for a in ACTIONS)

        td_error = reward + self.gamma * best_next_q - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error

        # Accumulate reward for episode-level logging
        self._current_episode_reward += reward

    def end_episode(self) -> None:
        """
        Call once at the end of every training episode.
        Decays epsilon and records the episode's total reward.
        """
        self.episode += 1
        self.episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0

        # Decay epsilon, but never below the floor
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Serialise the Q-table and hyperparameters to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table":       self.q_table,
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "episode":       self.episode,
                "episode_rewards": self.episode_rewards,
            }, f)
        print(f"[QLearningAgent] Saved Q-table ({len(self.q_table)} entries) to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "QLearningAgent":
        """Deserialise a previously saved agent from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha         = data["alpha"],
            gamma         = data["gamma"],
            epsilon       = data["epsilon"],
            epsilon_min   = data["epsilon_min"],
            epsilon_decay = data["epsilon_decay"],
        )
        agent.q_table         = data["q_table"]
        agent.episode         = data["episode"]
        agent.episode_rewards = data["episode_rewards"]
        print(f"[QLearningAgent] Loaded Q-table ({len(agent.q_table)} entries) from {filepath}")
        return agent

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Print a short summary of training progress to the console."""
        if not self.episode_rewards:
            print("[QLearningAgent] No episodes completed yet.")
            return
        recent = self.episode_rewards[-10:]
        print(
            f"[QLearningAgent] Episodes: {self.episode} | "
            f"Q-table size: {len(self.q_table)} | "
            f"ε: {self.epsilon:.4f} | "
            f"Avg reward (last 10 ep): {sum(recent)/len(recent):.1f}"
        )