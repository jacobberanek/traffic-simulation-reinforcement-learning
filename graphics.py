import tkinter
import math


class Graphics:
    """Tkinter-based renderer for the traffic intersection simulator."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.root = tkinter.Tk()
        self.root.title("Traffic Simulation")
        self.canvas = tkinter.Canvas(self.root, width=width, height=height)
        self.canvas.pack()

    def draw_rectangle(self, x1, y1, x2, y2, color):
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def draw_circle(self, x, y, radius, color):
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

    def draw_line(self, x1, y1, x2, y2, color):
        self.canvas.create_line(x1, y1, x2, y2, fill=color)

    def _rotate_point(self, px, py, cx, cy, angle):
        """Rotate a point (px, py) around center (cx, cy) by angle (in degrees)."""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x = (px - cx) * cos_a - (py - cy) * sin_a + cx
        y = (px - cx) * sin_a + (py - cy) * cos_a + cy
        return x, y

    def draw_car(self, x, y, color, text, rotation=0):
        # Center of the car for rotation
        cx, cy = x + 25, y + 12.5
        
        if rotation == 0:
            self.draw_rectangle(x, y, x + 50, y + 25, color)
            self.draw_circle(x + 10, y + 25, 5, "black")
            self.draw_circle(x + 40, y + 25, 5, "black")
        else:
            # Rotate vehicle body and wheels
            corners = [
                self._rotate_point(x, y, cx, cy, rotation),
                self._rotate_point(x + 50, y, cx, cy, rotation),
                self._rotate_point(x + 50, y + 25, cx, cy, rotation),
                self._rotate_point(x, y + 25, cx, cy, rotation),
            ]
            flat_coords = [coord for point in corners for coord in point]
            self.canvas.create_polygon(flat_coords, fill=color, outline="black")
            
            # Rotate wheels
            wheel1 = self._rotate_point(x + 10, y + 25, cx, cy, rotation)
            wheel2 = self._rotate_point(x + 40, y + 25, cx, cy, rotation)
            self.draw_circle(int(wheel1[0]), int(wheel1[1]), 5, "black")
            self.draw_circle(int(wheel2[0]), int(wheel2[1]), 5, "black")
        
        self.canvas.create_text(int(cx), int(cy), text=text)

    def draw_truck(self, x, y, color, text, rotation=0):
        # Center of the truck for rotation
        cx, cy = x + 35, y + 15
        
        if rotation == 0:
            self.draw_rectangle(x, y, x + 70, y + 30, color)
            self.draw_rectangle(x + 50, y, x + 70, y + 30, color)
            self.draw_circle(x + 10, y + 30, 5, "black")
            self.draw_circle(x + 40, y + 30, 5, "black")
            self.draw_circle(x + 60, y + 30, 5, "black")
        else:
            # Rotate truck body
            cabin_corners = [
                self._rotate_point(x, y, cx, cy, rotation),
                self._rotate_point(x + 70, y, cx, cy, rotation),
                self._rotate_point(x + 70, y + 30, cx, cy, rotation),
                self._rotate_point(x, y + 30, cx, cy, rotation),
            ]
            flat_coords = [coord for point in cabin_corners for coord in point]
            self.canvas.create_polygon(flat_coords, fill=color, outline="black")

            # Rotate cab
            bed_corners = [
                self._rotate_point(x + 50, y, cx, cy, rotation),
                self._rotate_point(x + 70, y, cx, cy, rotation),
                self._rotate_point(x + 70, y + 30, cx, cy, rotation),
                self._rotate_point(x + 50, y + 30, cx, cy, rotation),
            ]
            flat_coords = [coord for point in bed_corners for coord in point]
            self.canvas.create_polygon(flat_coords, fill=color, outline="black")
            
            # Rotate wheels
            wheel1 = self._rotate_point(x + 10, y + 30, cx, cy, rotation)
            wheel2 = self._rotate_point(x + 40, y + 30, cx, cy, rotation)
            wheel3 = self._rotate_point(x + 60, y + 30, cx, cy, rotation)
            self.draw_circle(int(wheel1[0]), int(wheel1[1]), 5, "black")
            self.draw_circle(int(wheel2[0]), int(wheel2[1]), 5, "black")
            self.draw_circle(int(wheel3[0]), int(wheel3[1]), 5, "black")
        
        self.canvas.create_text(int(cx), int(cy), text=text)

    def draw_traffic_light(self, x, y, color, text=None):
        self.draw_rectangle(x, y, x + 20, y + 60, "gray")
        if color == "red":
            self.draw_circle(x + 10, y + 10, 8, "red")
            self.draw_circle(x + 10, y + 30, 8, "black")
            self.draw_circle(x + 10, y + 50, 8, "black")
        elif color == "yellow":
            self.draw_circle(x + 10, y + 10, 8, "black")
            self.draw_circle(x + 10, y + 30, 8, "yellow")
            self.draw_circle(x + 10, y + 50, 8, "black")
        elif color == "green":
            self.draw_circle(x + 10, y + 10, 8, "black")
            self.draw_circle(x + 10, y + 30, 8, "black")
            self.draw_circle(x + 10, y + 50, 8, "green")
        if text:
            self.canvas.create_text(x + 10, y + 70, text=text)

    def draw_road(self, x1, y1, x2, y2, north_south=False):
        if north_south:
            self.draw_rectangle(x1, y1, x2, y2, "gray")
            for i in range(y1 + 10, y2, 20):
                self.draw_rectangle((x1 + x2) // 2 - 5, i, (x1 + x2) // 2 + 5, i + 10, "white")
        if not north_south:
            self.draw_rectangle(x1, y1, x2, y2, "gray")
            for i in range(x1 + 10, x2, 20):
                self.draw_rectangle(i, (y1 + y2) // 2 - 5, i + 10, (y1 + y2) // 2 + 5, "white")

    def draw_four_way_intersection(self):
        x1 = int(self.width * 0.4)
        x2 = int(self.width * 0.6)
        y1 = int(self.height * 0.4)
        y2 = int(self.height * 0.6)

        self.draw_road(x1, 0, x2, self.height, north_south=True)
        self.draw_road(0, y1, self.width, y2, north_south=False)

        self.draw_rectangle(x1, y1, x2, y2, "gray")

        self.draw_traffic_light(int(self.width * 0.24), int(self.height * 0.24), "red", "N")
        self.draw_traffic_light(int(self.width * 0.76), int(self.height * 0.24), "red", "E")
        self.draw_traffic_light(int(self.width * 0.24), int(self.height * 0.76), "red", "W")
        self.draw_traffic_light(int(self.width * 0.76), int(self.height * 0.76), "red", "S")
        


    def clear(self):
        self.canvas.delete("all")

    def mainloop(self):
        tkinter.mainloop()


    # ── Helpers used by Simulator to drive the animation ─────────────────────

    def after(self, ms: int, callback) -> None:
        self.root.after(ms, callback)

    def draw_traffic_lights_for_state(self, light_state_value: str) -> None:
        """Draw all four traffic lights based on the current LightState value string."""

        if light_state_value == "NS_GREEN":
            ns_color, ew_color = "green", "red"
        elif light_state_value == "NS_YELLOW":
            ns_color, ew_color = "yellow", "red"
        elif light_state_value == "EW_GREEN":
            ns_color, ew_color = "red", "green"
        elif light_state_value == "EW_YELLOW":
            ns_color, ew_color = "red", "yellow"
        else:
            ns_color, ew_color = "red", "red"
        
        self.draw_traffic_light(int(self.width * 0.24), int(self.height * 0.24), ns_color, "N")
        self.draw_traffic_light(int(self.width * 0.76), int(self.height * 0.24), ew_color, "E")
        self.draw_traffic_light(int(self.width * 0.24), int(self.height * 0.76), ew_color, "W")
        self.draw_traffic_light(int(self.width * 0.76), int(self.height * 0.76), ns_color, "S")

    
    def draw_queues(self, queues):
        """
        Draw up to 5 cars per direction queue approaching the intersection.
        queues: dict[Direction -> CarQueue]
        """

        W, H = self.width, self.height
        spacing = 55  # pixels between cars in queue

        for direction, queue in queues.items():
            cars = list(queue)[:5]
            dir_name = direction.value

            for i, car in enumerate(cars):
                label = str(car.car_id)
                if dir_name == "N":
                    # Driving south, queued above intersection
                    cx = int(W * 0.41)
                    cy = int(H * 0.31) - i * spacing
                    self.draw_car(cx, cy, "deepskyblue", label, rotation=90)
                elif dir_name == "S":
                    # Driving north, queued below intersection
                    cx = int(W * 0.49)
                    cy = int(H * 0.63) + i * spacing
                    self.draw_car(cx, cy, "tomato", label, rotation=270)
                elif dir_name == "E":
                    # Driving west, queued right of intersection
                    cx = int(W * 0.63) + i * spacing
                    cy = int(H * 0.41)
                    self.draw_car(cx, cy, "gold", label, rotation=180)
                elif dir_name == "W":
                    # Driving east, queued left of intersection
                    cx = int(W * 0.31) - i * spacing
                    cy = int(H * 0.49)
                    self.draw_car(cx, cy, "limegreen", label, rotation=0)


if __name__ == "__main__":
    graphics = Graphics(500, 500)
    graphics.draw_four_way_intersection()

    graphics.draw_car(x=25, y=260, color="blue", text="East", rotation=0)
    graphics.draw_car(x=25, y=210, color="red", text="West", rotation=180)
    graphics.draw_truck(x=100, y=260, color="blue", text="East", rotation=0)
    graphics.draw_truck(x=100, y=210, color="red", text="West", rotation=180)
    
    graphics.draw_car(x=255, y=25, color="green", text="North", rotation=270)
    graphics.draw_car(x=200, y=25, color="yellow", text="South", rotation=90)
    graphics.draw_truck(x=255-10, y=100, color="green", text="North", rotation=270)
    graphics.draw_truck(x=200-10, y=100, color="yellow", text="South", rotation=90)

    # X: 255 = north, 200 = south (variable E/W) (subtract 10 for trucks to keep them centered on the road)
    # Y: 260 = east, 210 = west (variable N/S)
    # Rotation: 0 = east, 90 = south, 180 = west, 270 = north

    
    graphics.mainloop()