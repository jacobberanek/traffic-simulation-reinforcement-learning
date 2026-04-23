from collections.abc import Callable
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

    count = 0  # global counter; never reset so IDs are unique across episodes

    def __init__(self, origin: Direction, move: Move, created_tick: int):
        Car.count        += 1
        self.car_id       = Car.count
        self.origin       = origin
        self.move         = move
        self.created_tick = created_tick
        self.wait_time    = -1  # set by update_wait_time() when the car is processed

    def update_wait_time(self, current_tick: int) -> None:
        self.wait_time = current_tick - self.created_tick


class CarQueue(deque):
    """deque subclass that also tracks per-direction throughput statistics."""

    def __init__(self):
        super().__init__()
        self.avg_wait_time        = 0.0
        self.total_cars_processed = 0
        self.processed_car_queue  = deque()

    def add_car(self, car: Car) -> None:
        self.append(car)

    def move_car(self, current_tick: int) -> "Car | None":
        """Remove the front car, record its wait time, and return it."""
        if not self:
            return None
        self.total_cars_processed += 1
        wait = current_tick - self[0].created_tick
        self.avg_wait_time = (
            (self.avg_wait_time * (self.total_cars_processed - 1) + wait)
            / self.total_cars_processed
        )
        car = self.popleft()
        car.update_wait_time(current_tick)
        self.processed_car_queue.append(car)
        return car


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
        falls back to a fixed-timing schedule.
    fixed_switch_interval : int
        Tick interval used by the fixed-timing baseline (ignored when an
        agent is present).
    seed : int or None
        Random seed for reproducible runs.  None means non-deterministic.
    max_green_ticks : int
        Maximum ticks a green phase is held before a forced switch.
    traffic_pattern : Callable[[int], dict[Direction, int]] or None
        Returns per-direction spawn weights for a given tick.  Defaults to
        North-heavy traffic (4:1:1:1).
    spawn_interval : Callable[[int], int] or None
        Returns the number of ticks between spawn events.  Defaults to 3.
    spawn_count : Callable[[int], int] or None
        Returns how many cars to spawn per event.  Defaults to 2.
    """

    def __init__(
        self,
        duration:              int             = 200,
        delay:                 float           = 0.5,
        moves_per_tick:        int             = 1,
        agent:                 QLearningAgent  = None,
        fixed_switch_interval: int             = 10,
        seed:                  int | None      = 42,
        max_green_ticks:       int             = 30,
        traffic_pattern:       Callable | None = None,
        spawn_interval:        Callable | None = None,
        spawn_count:           Callable | None = None,
    ):
        self.duration              = duration
        self.delay                 = delay
        self.moves_per_tick        = moves_per_tick
        self.agent                 = agent
        self.fixed_switch_interval = fixed_switch_interval
        self.seed                  = seed
        self.max_green_ticks       = max_green_ticks
        self.current_green_ticks   = 0

        # _spawn_directions defines the canonical key order used when zipping
        # weights from traffic_pattern() into random.choices().
        self._spawn_directions = list(Direction)

        if traffic_pattern is not None:
            self.traffic_pattern = traffic_pattern
        else:
            self.traffic_pattern = lambda tick: {
                Direction.NORTH: 4,
                Direction.SOUTH: 1,
                Direction.EAST:  1,
                Direction.WEST:  1,
            }

        if spawn_interval is not None:
            self.spawn_interval = spawn_interval
        else:
            self.spawn_interval = lambda tick: 3

        if spawn_count is not None:
            self.spawn_count = spawn_count
        else:
            self.spawn_count = lambda tick: 2

        self._rng = random.Random(seed)

        self.time                 = 0
        self.yellow_delay         = 3  # ticks the yellow phase lasts
        self.yellow_time_remaining = 0

        self.queues = {
            Direction.NORTH: CarQueue(),
            Direction.SOUTH: CarQueue(),
            Direction.EAST:  CarQueue(),
            Direction.WEST:  CarQueue(),
        }

        self.light_state = LightState.NS_GREEN

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset the simulator back to tick 0 so the same agent can be trained
        across multiple episodes without constructing a new object.

        Note: Car.count is NOT reset so car IDs remain globally unique across
        episodes, which is useful for debugging.
        """
        self.time                 = 0
        self.yellow_time_remaining = 0
        self.light_state          = LightState.NS_GREEN
        self._rng                 = random.Random(self.seed)
        self.current_green_ticks  = 0

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
            self.light_state,
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _weighted_queue_cost(self, directions: list) -> float:
        """Sum exponential wait penalties across all cars in the given directions.

        Each car contributes e^(wait / 50), so a car waiting 50 ticks costs
        ~2.7x more than a freshly arrived car.
        """
        total = 0.0
        for d in directions:
            for car in self.queues[d]:
                wait = self.time - car.created_tick
                total += math.exp(wait / 50.0)
        return total

    def get_reward(self, switched: bool = False) -> float:
        ns_cost = self._weighted_queue_cost([Direction.NORTH, Direction.SOUTH])
        ew_cost = self._weighted_queue_cost([Direction.EAST,  Direction.WEST])

        queue_penalty = -(ns_cost + ew_cost)

        # Bonus for switching when the waiting direction has a long queue
        bonus = 0.0
        if switched:
            if self.light_state in [LightState.NS_GREEN, LightState.NS_YELLOW]:
                bonus = ew_cost * 2
            else:
                bonus = ns_cost * 2

        # Small fixed penalty to discourage unnecessary switching
        switch_penalty = -0.5 if switched else 0.0

        return queue_penalty + bonus + switch_penalty

    # ── Main run loop ─────────────────────────────────────────────────────────

    def _step(self) -> None:
        """Advance the simulation by one tick: spawn cars, process lights, decide, move."""
        self.time += 1

        if self.time % self.spawn_interval(self.time) == 0:
            pattern = self.traffic_pattern(self.time)
            for _ in range(self.spawn_count(self.time)):
                self.create_car(
                    self._rng.choices(
                        self._spawn_directions,
                        weights=[pattern[d] for d in self._spawn_directions],
                    )[0],
                    self._rng.choice(list(Move)),
                )

        self.process_light()

        if self.agent is not None:
            state    = self.get_state()
            action   = self.agent.choose_action(state)
            switched = False
            if action == SWITCH or self.current_green_ticks >= self.max_green_ticks:
                switched = self.trigger_light_switch()
                if switched:
                    self.current_green_ticks = 0
            if self.light_state in [LightState.NS_GREEN, LightState.EW_GREEN]:
                self.current_green_ticks += 1
            self.move_cars()
            reward     = self.get_reward(switched)
            next_state = self.get_state()
            self.agent.update(state, action, reward, next_state)
        else:
            if self.time % self.fixed_switch_interval == 0:
                self.trigger_light_switch()
            self.move_cars()

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
            self._step()
            if verbose:
                self.print_state()
                time.sleep(self.delay)

        stats = self._collect_stats()
        if verbose:
            self._print_summary(stats)
        return stats

    # ── Tkinter animation tick ────────────────────────────────────────────────

    def _tick(self) -> None:
        """Single tick step, scheduled repeatedly via Tkinter after()."""
        if self.time >= self.duration:
            stats = self._collect_stats()
            self._print_summary(stats)
            return  # stop scheduling — simulation is done

        self._step()
        self.print_state()

        self.graphics.clear()
        self.graphics.draw_four_way_intersection()
        self.graphics.draw_traffic_lights_for_state(self.light_state.value)
        self.graphics.draw_queues(self.queues)

        # Schedule next tick (delay converted from seconds to milliseconds)
        self.graphics.after(int(self.delay * 1000), self._tick)

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def _collect_stats(self) -> dict:
        """Return a dictionary of episode statistics."""
        total_processed = sum(q.total_cars_processed for q in self.queues.values())
        total_remaining = sum(len(q) for q in self.queues.values())

        if total_processed > 0:
            avg_wait = sum(
                q.avg_wait_time * q.total_cars_processed for q in self.queues.values()
            ) / total_processed
        else:
            avg_wait = 0.0

        per_direction = {
            d.value: (q.avg_wait_time if q.total_cars_processed > 0 else 0.0)
            for d, q in self.queues.items()
        }

        longest = sorted(
            (car for q in self.queues.values() for car in q.processed_car_queue),
            key=lambda c: c.wait_time,
            reverse=True,
        )[:5]

        return {
            "total_processed": total_processed,
            "total_remaining": total_remaining,
            "avg_wait":        avg_wait,
            "per_direction":   per_direction,
            "longest_waits":   [(c.car_id, c.wait_time) for c in longest],
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

    def print_state(self) -> None:
        print(f"Time: {self.time}")
        print(f"Light State: {self.light_state.value}")
        if self.light_state in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            print(f"Yellow time remaining: {self.yellow_time_remaining}")
        for direction, queue in self.queues.items():
            print(f"{direction.value}: {[car.car_id for car in queue]}")

    # ── Car / light helpers ───────────────────────────────────────────────────

    def create_car(self, origin: Direction, move: Move) -> None:
        self.queues[origin].add_car(Car(origin, move, self.time))

    def process_light(self) -> None:
        """Must run every tick — handles the yellow-phase countdown."""
        if self.light_state in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            if self.yellow_time_remaining > 0:
                self.yellow_time_remaining -= 1
            if self.yellow_time_remaining == 0:
                self.switch_light()

    def trigger_light_switch(self) -> bool:
        """
        Request a light phase change.  Returns False and does nothing if
        the light is currently in a yellow phase (the agent must wait).
        """
        if self.light_state in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            return False
        self.switch_light()
        return True

    def switch_light(self) -> None:
        """Advance the light state machine by one step."""
        if self.light_state == LightState.NS_GREEN:
            self.light_state          = LightState.NS_YELLOW
            self.yellow_time_remaining = self.yellow_delay
        elif self.light_state == LightState.NS_YELLOW:
            self.light_state = LightState.EW_GREEN
        elif self.light_state == LightState.EW_GREEN:
            self.light_state          = LightState.EW_YELLOW
            self.yellow_time_remaining = self.yellow_delay
        elif self.light_state == LightState.EW_YELLOW:
            self.light_state = LightState.NS_GREEN

    def move_cars(self) -> None:
        """Clear moves_per_tick cars per green direction each tick."""
        for _ in range(self.moves_per_tick):
            if self.light_state == LightState.NS_GREEN:
                for direction in [Direction.NORTH, Direction.SOUTH]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)
            elif self.light_state == LightState.EW_GREEN:
                for direction in [Direction.EAST, Direction.WEST]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    simulator = Simulator()
    simulator.run(visual=True)
