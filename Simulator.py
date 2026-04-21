import math
import random
import time
from collections import deque
from enum import Enum

from QLearningAgent import QLearningAgent, bucket, KEEP, SWITCH


class Direction(Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST  = "E"
    WEST  = "W"

class Move(Enum):
    THROUGH = "THROUGH"
    RIGHT   = "RIGHT"

class LightState(Enum):
    NS_GREEN  = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    EW_GREEN  = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"


class Car:

    count    = 0
    waitTime = -1  # Placeholder until car is processed

    def __init__(self, origin, move, createdTick):
        Car.count      += 1
        self.carID      = Car.count
        self.origin     = origin
        self.move       = move
        self.createdTick = createdTick

    def update_wait_time(self, currentTick):
        self.waitTime = currentTick - self.createdTick


# Created to efficiently track secondary stats for each queue
class CarQueue(deque):
    def __init__(self):
        super().__init__()
        self.avgWaitTime       = 0
        self.totalCarsProcessed = 0
        self.processedCarQueue  = deque()  # Queue to track processed cars for final stats

    def add_car(self, car):
        self.append(car)

    def move_car(self, currentTick):
        if self:
            self.totalCarsProcessed += 1
            waitTime = currentTick - self[0].createdTick
            self.avgWaitTime = (
                (self.avgWaitTime * (self.totalCarsProcessed - 1) + waitTime)
                / self.totalCarsProcessed
            )
            processed_car = self.popleft()
            processed_car.update_wait_time(currentTick)
            self.processedCarQueue.append(processed_car)
            return processed_car
        return None


class Simulator:
    """
    Discrete-time traffic intersection simulator.

    Parameters
    ----------
    duration : int
        Number of ticks to run per episode.
    delay : float
        Seconds to sleep between ticks when running with visuals or in
        human-readable mode.  Set to 0 for fast headless training.
    moves_per_tick : int
        How many cars the green-light directions can clear each tick.
    agent : QLearningAgent or None
        When provided the agent controls the light; otherwise the simulator
        falls back to a fixed 10-tick switching schedule (the baseline).
    fixed_switch_interval : int
        Tick interval used by the fixed-timing baseline (ignored when an
        agent is present).
    seed : int or None
        Random seed for reproducible runs.  None means non-deterministic.
    """

    def __init__(
        self,
        duration:              int             = 200,
        delay:                 float           = 0.5,
        moves_per_tick:        int             = 1,
        agent:                 QLearningAgent  = None,
        fixed_switch_interval: int             = 10,
        seed:                  int | None      = 42, #Set to a trivial number to keep results consistent
        max_green_ticks:       int             = 30,  # force switch if green held longer than this
        spawn_weights:         dict | None     = None,
    ):
        self.duration              = duration
        self.delay                 = delay
        self.moves_per_tick        = moves_per_tick
        self.agent                 = agent
        self.fixed_switch_interval = fixed_switch_interval
        self.seed                  = seed
        self.max_green_ticks       = max_green_ticks
        self.current_green_ticks   = 0

        # Weighted car spawning — default gives North 4x more traffic than other directions.
        # Pass a dict like {Direction.NORTH: 4, Direction.SOUTH: 1, ...} to override.
        default_weights = {
            Direction.NORTH: 4,
            Direction.SOUTH: 1,
            Direction.EAST:  1,
            Direction.WEST:  1,
        }
        weights = spawn_weights if spawn_weights is not None else default_weights
        self._spawn_directions = list(weights.keys())
        self._spawn_weights    = list(weights.values())

        self._rng = random.Random(seed)  # seeded RNG for reproducibility

        self.time                = 0
        self.yellowDelay         = 3  # how many ticks the yellow light lasts
        self.yellowTimeRemaining = 0  # active counter for yellow phase

        self.queues = {
            Direction.NORTH: CarQueue(),
            Direction.SOUTH: CarQueue(),
            Direction.EAST:  CarQueue(),
            Direction.WEST:  CarQueue(),
        }

        self.lightState = LightState.NS_GREEN  # default starting state

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset the simulator back to tick 0 so the same agent can be trained
        across multiple episodes without constructing a new object.

        Note: Car.count is NOT reset so car IDs remain globally unique across
        episodes, which is useful for debugging.
        """
        self.time                = 0
        self.yellowTimeRemaining = 0
        self.lightState          = LightState.NS_GREEN
        self._rng                = random.Random(self.seed)  # re-seed for reproducibility
        self.current_green_ticks = 0

        self.queues = {
            Direction.NORTH: CarQueue(),
            Direction.SOUTH: CarQueue(),
            Direction.EAST:  CarQueue(),
            Direction.WEST:  CarQueue(),
        }

    # ── State observation ─────────────────────────────────────────────────────

    def get_state(self) -> tuple:
        """
        Return a discretised snapshot of the intersection as a tuple:

            (bucketed_N, bucketed_S, bucketed_E, bucketed_W, light_state)

        This is what the agent observes each tick.  Queue lengths are bucketed
        (see QLearningAgent.bucket) to keep the state space small.
        """
        return (
            bucket(len(self.queues[Direction.NORTH])),
            bucket(len(self.queues[Direction.SOUTH])),
            bucket(len(self.queues[Direction.EAST])),
            bucket(len(self.queues[Direction.WEST])),
            self.lightState,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _weighted_queue_cost(self, directions: list) -> float:
        """Sum exponential wait penalties across all cars in the given directions.

        Each car contributes e^(wait_ticks / 20), so a car waiting 20 ticks
        costs ~2.7x more than a freshly arrived car, and one waiting 60 ticks
        costs ~20x more.  The /20 scale factor keeps values manageable over a
        300-tick episode.
        """
        total = 0.0
        for d in directions:
            for car in self.queues[d]:
                wait = self.time - car.createdTick
                total += math.exp(wait / 20.0) # TODO: Tune this if needed so that the value doesn't blow up TOO fast
        return total

    def get_reward(self, switched: bool = False) -> float:
        ns_cost = self._weighted_queue_cost([Direction.NORTH, Direction.SOUTH])
        ew_cost = self._weighted_queue_cost([Direction.EAST,  Direction.WEST])

        queue_penalty = -(ns_cost + ew_cost)

        # reward switching when the waiting direction is backed up
        bonus = 0.0
        if switched:
            if self.lightState in [LightState.NS_GREEN, LightState.NS_YELLOW]:
                bonus = ew_cost * 2   # switching to EW is good when EW is backed up
            else:
                bonus = ns_cost * 2   # switching to NS is good when NS is backed up

        return queue_penalty + bonus

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, visual: bool = False, verbose: bool = True) -> dict:
        """
        Run one full episode (self.duration ticks).

        Parameters
        ----------
        visual  : bool  — open the Tkinter window and animate.
        verbose : bool  — print per-tick state to console.

        Returns
        -------
        dict with summary statistics for this episode (used by the training
        loop to collect results without parsing printed output).
        """

        if visual:
            from graphics import Graphics
            self.graphics = Graphics(500, 500)
            self.graphics.after(0, self._tick)
            self.graphics.mainloop()
            # Stats are printed inside _print_summary; return empty dict for
            # visual runs since the Tkinter loop is blocking.
            return {}

        while self.time < self.duration:
            self.time += 1

            # ── Spawn cars ────────────────────────────────────────────────
            if self.time % 3 == 0:
                for _ in range(self._rng.randint(1, 3)):
                    self.create_car(
                        self._rng.choices(self._spawn_directions, weights=self._spawn_weights)[0],
                        self._rng.choice(list(Move)),
                    )

            # ── Light processing (yellow countdown) ───────────────────────
            self.process_light()

            # ── Agent or fixed-timing decision ────────────────────────────
            if self.agent is not None:
                state  = self.get_state()
                action = self.agent.choose_action(state)

                switched = False
                if action == SWITCH or self.current_green_ticks >= self.max_green_ticks:
                    switched = self.trigger_light_switch()
                    if switched:
                        self.current_green_ticks = 0

                if self.lightState in [LightState.NS_GREEN, LightState.EW_GREEN]:
                    self.current_green_ticks += 1

                self.move_cars()

                reward     = self.get_reward(switched)
                next_state = self.get_state()
                self.agent.update(state, action, reward, next_state)

            else:
                # Fixed-timing baseline: switch every fixed_switch_interval ticks
                if self.time % self.fixed_switch_interval == 0:
                    self.trigger_light_switch()
                self.move_cars()

            if verbose:
                self.print_state()
                time.sleep(self.delay)

        stats = self._collect_stats()
        if verbose:
            self._print_summary(stats)
        return stats

    # ── Tkinter animation tick ────────────────────────────────────────────────

    def _tick(self):
        """Single tick step, scheduled repeatedly via Tkinter after()."""
        if self.time >= self.duration:
            stats = self._collect_stats()
            self._print_summary(stats)
            return  # stop scheduling — simulation is done

        self.time += 1

        # Spawn cars
        if self.time % 3 == 0:
            for _ in range(self._rng.randint(1, 3)):
                self.create_car(
                    self._rng.choices(self._spawn_directions, weights=self._spawn_weights)[0],
                    self._rng.choice(list(Move)),
                )

        # Light processing
        self.process_light()

        # Agent or fixed-timing decision
        if self.agent is not None:
            state  = self.get_state()
            action = self.agent.choose_action(state)
            switched = False
            if action == SWITCH or self.current_green_ticks >= self.max_green_ticks:
                switched = self.trigger_light_switch()
                if switched:
                    self.current_green_ticks = 0
            
            if self.lightState in [LightState.NS_GREEN, LightState.EW_GREEN]:
                self.current_green_ticks += 1
            self.move_cars()
            reward     = self.get_reward(switched)
            next_state = self.get_state()
            self.agent.update(state, action, reward, next_state)
        else:
            if self.time % self.fixed_switch_interval == 0:
                self.trigger_light_switch()
            self.move_cars()

        self.print_state()

        # Draw the current state
        self.graphics.clear()
        self.graphics.draw_four_way_intersection()
        self.graphics.draw_traffic_lights_for_state(self.lightState.value)
        self.graphics.draw_queues(self.queues)

        # Schedule next tick (delay converted from seconds to milliseconds)
        self.graphics.after(int(self.delay * 1000), self._tick)

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def _collect_stats(self) -> dict:
        """Return a dictionary of episode statistics."""
        total_processed = sum(q.totalCarsProcessed for q in self.queues.values())
        total_remaining = sum(len(q) for q in self.queues.values())

        if total_processed > 0:
            avg_wait = sum(
                q.avgWaitTime * q.totalCarsProcessed for q in self.queues.values()
            ) / total_processed
        else:
            avg_wait = 0.0

        per_direction = {
            d.value: (q.avgWaitTime if q.totalCarsProcessed > 0 else 0.0)
            for d, q in self.queues.items()
        }

        longest = sorted(
            (car for q in self.queues.values() for car in q.processedCarQueue),
            key=lambda c: c.waitTime,
            reverse=True,
        )[:5]

        return {
            "total_processed": total_processed,
            "total_remaining": total_remaining,
            "avg_wait":        avg_wait,
            "per_direction":   per_direction,
            "longest_waits":   [(c.carID, c.waitTime) for c in longest],
        }

    def _print_summary(self, stats: dict) -> None:
        print("-" * 72)
        print("Simulation complete.")
        print(f"Total cars processed: {stats['total_processed']}")
        print(f"Total cars remaining in queues: {stats['total_remaining']}")
        print(f"Average wait time per car: {stats['avg_wait']:.2f} ticks")
        print("Average wait time per direction:")
        for d, w in stats["per_direction"].items():
            print(f"  {d}: {w:.2f} ticks")
        print("Car IDs with longest wait times:")
        if stats["longest_waits"]:
            for car_id, wait in stats["longest_waits"]:
                print(f"  Car ID {car_id}: {wait} ticks")
        else:
            print("  None")

    def print_state(self):
        print(f"Time: {self.time}")
        print(f"Light State: {self.lightState.value}")
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            print("Current Yellow Time Remaining: ", self.yellowTimeRemaining)
        for direction, queue in self.queues.items():
            print(f"{direction.value}: {[car.carID for car in queue]}")

    # ── Car / light helpers ───────────────────────────────────────────────────

    def create_car(self, origin, move):
        car = Car(origin, move, self.time)
        self.queues[origin].add_car(car)

    def process_light(self):
        """Must run every tick — handles the yellow-phase countdown."""
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            if self.yellowTimeRemaining > 0:
                self.yellowTimeRemaining -= 1
            if self.yellowTimeRemaining == 0:
                self.switch_light()

    def trigger_light_switch(self) -> bool:
        """
        Request a light phase change.  Returns False and does nothing if
        the light is currently in a yellow phase (the agent must wait).
        """
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            return False
        self.switch_light()
        return True

    def switch_light(self):
        """Advance the light state machine by one step."""
        if self.lightState == LightState.NS_GREEN:
            self.lightState          = LightState.NS_YELLOW
            self.yellowTimeRemaining = self.yellowDelay
        elif self.lightState == LightState.NS_YELLOW:
            self.lightState = LightState.EW_GREEN
        elif self.lightState == LightState.EW_GREEN:
            self.lightState          = LightState.EW_YELLOW
            self.yellowTimeRemaining = self.yellowDelay
        elif self.lightState == LightState.EW_YELLOW:
            self.lightState = LightState.NS_GREEN

    def move_cars(self):
        """Clear one car per green-direction per tick (or moves_per_tick cars)."""
        for _ in range(self.moves_per_tick):
            if self.lightState == LightState.NS_GREEN:
                for direction in [Direction.NORTH, Direction.SOUTH]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)
            elif self.lightState == LightState.EW_GREEN:
                for direction in [Direction.EAST, Direction.WEST]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    simulator = Simulator()
    simulator.run(visual=True)