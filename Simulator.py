import random
import time
from collections import deque
from enum import Enum

class Direction(Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"

class Move(Enum):
    THROUGH = "THROUGH"
    RIGHT = "RIGHT"

class LightState(Enum):
    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"

class Car:

    count = 0
    waitTime = -1 # Placeholder until car is processed

    def __init__(self, origin, move, createdTick):
        Car.count += 1
        self.carID = Car.count
        self.origin = origin
        self.move = move
        self.createdTick = createdTick

    def update_wait_time(self, currentTick):
        self.waitTime = currentTick - self.createdTick

# Created to efficiently track secondary stats for each queue
class CarQueue(deque):
    def __init__(self):
        super().__init__()
        self.avgWaitTime = 0
        self.totalCarsProcessed = 0
        self.processedCarQueue = deque() # Queue to track processed cars for final stats

    def add_car(self, car):
        self.append(car)

    def move_car(self, currentTick):
        if self:
            self.totalCarsProcessed += 1
            waitTime = currentTick - self[0].createdTick
            self.avgWaitTime = (self.avgWaitTime * (self.totalCarsProcessed - 1) + waitTime) / self.totalCarsProcessed
            processed_car = self.popleft()
            processed_car.update_wait_time(currentTick)
            self.processedCarQueue.append(processed_car)
            return processed_car
        return None

class Simulator:
    def __init__(
            self,
            duration: int = 200,
            delay: float = 0.5,
            moves_per_tick: int = 1
        ):
         
        self.duration = duration
        self.delay = delay
        self.moves_per_tick = moves_per_tick

        self.time = 0
        self.yellowDelay = 3 # how many ticks the yellow light lasts
        self.yellowTimeRemaining = 0 # active counter for yellow phase

        self.queues = {
            Direction.NORTH: CarQueue(),
            Direction.SOUTH: CarQueue(),
            Direction.EAST: CarQueue(),
            Direction.WEST: CarQueue()
        }

        self.lightState = LightState.NS_GREEN #default starting state

    def run(self, visual=False):

        #Visualization can be toggled on or off when running simulation
        if visual:
            from graphics import Graphics
            self.graphics = Graphics(500, 500)
            self.graphics.after(0, self._tick)
            self.graphics.mainloop()
            return

        while self.time < self.duration:
            self.time += 1
            
            # For now, randomly create 1-3 cars every 3 ticks (with random direction and move)
            if self.time % 3 == 0:
                for _ in range(random.randint(1, 3)):
                    self.create_car(random.choice(list(Direction)), random.choice(list(Move)))

            self.process_light()

            # For now, switch light every 10 ticks
            if self.time % 10 == 0:
                self.trigger_light_switch()

            self.move_cars()

            self.print_state()
            time.sleep(self.delay)
        
        self._print_summary()

    #Single tick step, scheduled repeatedly via Tkinter after()
    def _tick(self):
        if self.time >= self.duration:
            self._print_summary()
            return #stop scheduling - simulation is done
        
        self.time +=1

        if self.time % 3 == 0:
            for _ in range(random.randint(1,3)):
                self.create_car(random.choice(list(Direction)), random.choice(list(Move)))
        
        self.process_light()
        
        #For now, switch light every 10 ticks
        if self.time % 10 == 0:
            self.trigger_light_switch()

        self.move_cars()

        self.print_state()

        #Draw the current state
        self.graphics.clear()
        self.graphics.draw_four_way_intersection()
        self.graphics.draw_traffic_lights_for_state(self.lightState.value)
        self.graphics.draw_queues(self.queues)

        #Schedule next tick (delay converted from seconds to milliseconds)
        self.graphics.after(int(self.delay * 1000), self._tick)
    
    def _print_summary(self):
        print("-" * 72)
        print("Simulation complete.")

        total_cars_processed = sum(queue.totalCarsProcessed for queue in self.queues.values())
        print(f"Total cars processed: {total_cars_processed}")

        print(f"Total cars remaining in queues: {sum(len(queue) for queue in self.queues.values())}")

        if total_cars_processed > 0:
            avg_wait_time = sum(
                queue.avgWaitTime * queue.totalCarsProcessed for queue in self.queues.values()
            ) / total_cars_processed
            print(f"Average wait time per car: {avg_wait_time:.2f} ticks")
        else:
            print("Average wait time per car: 0.00 ticks")

        print("Average wait time per direction:")
        for direction, queue in self.queues.items():
            if queue.totalCarsProcessed > 0:
                print(f"  {direction.value}: {queue.avgWaitTime:.2f} ticks")
            else:
                print(f"  {direction.value}: 0.00 ticks")

        print("Car IDs with longest wait times:")
        longest_wait_cars = sorted(
            (car for queue in self.queues.values() for car in queue.processedCarQueue),
            key=lambda c: c.waitTime,
            reverse=True
        )[:5]
        if longest_wait_cars:
            for car in longest_wait_cars:
                print(f"  Car ID {car.carID}: {car.waitTime} ticks")
        else:
            print("None")

    def print_state(self):
        print(f"Time: {self.time}")
        print(f"Light State: {self.lightState.value}")
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            print("Current Yellow Time Remaining: ", self.yellowTimeRemaining)
        for direction, queue in self.queues.items():
            print(f"{direction.value}: {[car.carID for car in queue]}")

    def create_car(self, origin, move):
        car = Car(origin, move, self.time)
        self.queues[origin].add_car(car)

    def process_light(self): # THIS NEEDS TO RUN EVERY TICK
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]: # If currently in yellow phase
            if self.yellowTimeRemaining > 0: # Decrement timer if still yellow
                self.yellowTimeRemaining -= 1

            if self.yellowTimeRemaining == 0: # Switch light if yellow phase just ended
                self.switch_light()

    def trigger_light_switch(self): # This is called by the user/agent to signal a switch, but the light will only switch if not yellow
        if self.lightState in [LightState.NS_YELLOW, LightState.EW_YELLOW]:
            return False # Light is currently yellow, cannot switch yet. Can add special handling later to let agent know that the switch failed

        # Switch light immediately if green (no need to be in phase)
        self.switch_light()
        return True

    def switch_light(self):
        # Light is able to switch
        if self.lightState == LightState.NS_GREEN:
            self.lightState = LightState.NS_YELLOW
            self.yellowTimeRemaining = self.yellowDelay # Reset yellow light timer
        elif self.lightState == LightState.NS_YELLOW:
            self.lightState = LightState.EW_GREEN
        elif self.lightState == LightState.EW_GREEN:
            self.lightState = LightState.EW_YELLOW
            self.yellowTimeRemaining = self.yellowDelay # Reset yellow light timer
        elif self.lightState == LightState.EW_YELLOW:
            self.lightState = LightState.NS_GREEN

    def move_cars(self):
        # For now, just move one car from the green light direction
        for _ in range(self.moves_per_tick):
            if self.lightState == LightState.NS_GREEN: # Process north/south queues
                for direction in [Direction.NORTH, Direction.SOUTH]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)
            elif self.lightState == LightState.EW_GREEN: # Process east/west queues
                for direction in [Direction.EAST, Direction.WEST]:
                    if self.queues[direction]:
                        self.queues[direction].move_car(self.time)

if __name__ == "__main__":
    simulator = Simulator()
    simulator.run(visual = True)