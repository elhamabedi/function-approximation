import random
import copy
import pickle
import tkinter as tk
import time

# تنظیمات بازی
BOARD_WIDTH = 15
BOARD_HEIGHT = 10
MAX_TURNS = 30

# نمادها
EMPTY = "."
AIRCRAFT = "A"
DRONE = "S"
MOUNTAIN = "M"
RELOAD = "R"
GOAL = "G"

NUM_DRONES = 4
NUM_MOUNTAINS = 10

ALPHA = 0.01
ALPHA_DRONE = 0.02

# اسامی ویژگی‌های هواپیما
aircraft_feature_names = [
    "Bias",
    "Distance to Goal (norm)",
    "Rockets (norm)",
    "Min Drone Distance (norm)",
    "Drones within 1 block",
    "Drones within 2 blocks",
    "Distance to Reload (norm)",
    "Min Mountain Distance (norm)",
    "Turn Ratio",
    "Avg Drone Distance (norm)",
    "Need Reload",
    "All Drones Destroyed"
]

# اسامی ویژگی‌های پهباد
drone_feature_names = [
    "Bias",
    "Distance to Aircraft (norm)",
    "Distance to Goal (norm)",
    "Distance to Nearest Drone (norm)",
    "Aircraft in Sight",
    "Nearby Drones Count",
    "Distance to Mountain (norm)",
    "Turn Ratio",
    "Is Surrounding Aircraft"
]

CELL_SIZE = 40  # اندازه هر سلول به پیکسل

class Game:
    def __init__(self, graphical=False):
        self.graphical = graphical
        self.reset_game()

    def reset_game(self):
        self.board = [[EMPTY for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.goal_pos = (BOARD_WIDTH - 1, BOARD_HEIGHT - 1)
        self._place_item(self.goal_pos, GOAL)
        self.aircraft_pos = self._get_random_empty_cell()
        self._place_item(self.aircraft_pos, AIRCRAFT)
        self.aircraft_rockets = 2
        self.drones = []
        for _ in range(NUM_DRONES):
            pos = self._get_random_empty_cell()
            self.drones.append(pos)
            self._place_item(pos, DRONE)
        self.mountains = []
        for _ in range(NUM_MOUNTAINS):
            pos = self._get_random_empty_cell()
            self.mountains.append(pos)
            self._place_item(pos, MOUNTAIN)
        self.reload_zone = self._get_random_empty_cell()
        self._place_item(self.reload_zone, RELOAD)
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.winner = None
        self.loss_reason = None
        self.W = [0.0 for _ in range(len(aircraft_feature_names))]
        self.drone_W = [0.0 for _ in range(len(drone_feature_names))]
        self.previous_drone_positions = [None] * NUM_DRONES
        self.drone_scores = [0] * NUM_DRONES
        self.drone_status = [True] * NUM_DRONES
        # ذخیره مثال‌های آموزشی جهت به‌روزرسانی وزن‌ها
        self.training_examples = []
        self.drone_training_examples = []

    def _get_random_empty_cell(self):
        while True:
            x = random.randint(0, BOARD_WIDTH - 1)
            y = random.randint(0, BOARD_HEIGHT - 1)
            if self.board[y][x] == EMPTY:
                return (x, y)

    def _place_item(self, pos, item):
        if pos is None:
            return
        x, y = pos
        self.board[y][x] = item

    def _clear_cell(self, pos):
        if pos is None:
            return
        x, y = pos
        self.board[y][x] = EMPTY

    def distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def euclidean_distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def is_path_clear(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        if x1 == x2 and y1 == y2:
            return True
        if x1 == x2:
            step = 1 if y2 > y1 else -1
            for y in range(y1 + step, y2, step):
                if (x1, y) in self.mountains:
                    return False
            return True
        if y1 == y2:
            step = 1 if x2 > x1 else -1
            for x in range(x1 + step, x2, step):
                if (x, y1) in self.mountains:
                    return False
            return True
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        x_step = dx / steps
        y_step = dy / steps
        for i in range(1, steps):
            x = round(x1 + i * x_step)
            y = round(y1 + i * y_step)
            if (x, y) in self.mountains:
                return False
        return True

    def is_aircraft_surrounded(self, pos):
        if pos is None:
            return False
        x, y = pos
        active_drones = [d for d in self.drones if d is not None]
        adjacent_positions = [
            (x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1] 
            if not (dx == 0 and dy == 0)
        ]
        adjacent_count = sum(1 for p in adjacent_positions if p in active_drones)
        if adjacent_count >= 2:
            return True
        main_directions = [(x, y+1), (x, y-1), (x-1, y), (x+1, y)]
        main_adjacent = [p for p in main_directions if p in active_drones]
        if len(main_adjacent) >= 4:
            return True
        adjacent_pairs = [
            [(x, y+1), (x-1, y)],
            [(x, y+1), (x+1, y)],
            [(x, y-1), (x-1, y)],
            [(x, y-1), (x+1, y)]
        ]
        return any(all(p in active_drones for p in pair) for pair in adjacent_pairs)

    def extract_features(self):
        features = []
        features.append(1.0)
        if self.aircraft_pos is not None:
            dist_goal = self.distance(self.aircraft_pos, self.goal_pos)
        else:
            dist_goal = (BOARD_WIDTH + BOARD_HEIGHT - 2)
        features.append(dist_goal / (BOARD_WIDTH + BOARD_HEIGHT - 2))
        features.append(self.aircraft_rockets / 3.0)
        if self.aircraft_pos is not None:
            drone_dists = [self.distance(self.aircraft_pos, d) for d in self.drones if d is not None]
            min_drone_dist = min(drone_dists) if drone_dists else (BOARD_WIDTH + BOARD_HEIGHT)
        else:
            min_drone_dist = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(min_drone_dist / (BOARD_WIDTH + BOARD_HEIGHT))
        if self.aircraft_pos is not None:
            count_within1 = sum(1 for d in self.drones 
                              if d is not None 
                              and self.euclidean_distance(self.aircraft_pos, d) <= 1.5)
            count_within2 = sum(1 for d in self.drones 
                              if d is not None 
                              and self.distance(self.aircraft_pos, d) <= 2)
        else:
            count_within1 = count_within2 = 0
        features.append(count_within1)
        features.append(count_within2)
        if self.aircraft_pos is not None:
            dist_reload = self.distance(self.aircraft_pos, self.reload_zone)
        else:
            dist_reload = (BOARD_WIDTH + BOARD_HEIGHT - 2)
        features.append(dist_reload / (BOARD_WIDTH + BOARD_HEIGHT - 2))
        if self.mountains and self.aircraft_pos is not None:
            mountain_dists = [self.distance(self.aircraft_pos, m) for m in self.mountains]
            min_mountain_dist = min(mountain_dists) if mountain_dists else (BOARD_WIDTH + BOARD_HEIGHT)
        else:
            min_mountain_dist = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(min_mountain_dist / (BOARD_WIDTH + BOARD_HEIGHT))
        features.append(self.turn / MAX_TURNS)
        if self.aircraft_pos is not None and self.drones:
            if drone_dists:
                avg_drone = sum(drone_dists) / len(drone_dists)
            else:
                avg_drone = (BOARD_WIDTH + BOARD_HEIGHT)
        else:
            avg_drone = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(avg_drone / (BOARD_WIDTH + BOARD_HEIGHT))
        features.append(1.0 if self.aircraft_rockets < 1 else 0.0)
        features.append(1.0 if all(d is None for d in self.drones) else 0.0)
        return features

    def drone_extract_features(self, drone_pos):
        features = []
        features.append(1.0)
        if self.aircraft_pos is not None:
            dist_aircraft = self.distance(drone_pos, self.aircraft_pos)
        else:
            dist_aircraft = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(dist_aircraft / (BOARD_WIDTH + BOARD_HEIGHT))
        dist_goal = self.distance(drone_pos, self.goal_pos)
        features.append(dist_goal / (BOARD_WIDTH + BOARD_HEIGHT))
        other_drones = [d for d in self.drones if d is not None and d != drone_pos]
        if other_drones:
            min_drone_dist = min([self.distance(drone_pos, d) for d in other_drones])
        else:
            min_drone_dist = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(min_drone_dist / (BOARD_WIDTH + BOARD_HEIGHT))
        features.append(1.0 if self.aircraft_pos is not None and dist_aircraft <= 3 else 0.0)
        nearby_drones = sum(1 for d in other_drones if self.distance(drone_pos, d) <= 2)
        features.append(nearby_drones / NUM_DRONES)
        if self.mountains:
            min_mountain_dist = min([self.distance(drone_pos, m) for m in self.mountains])
        else:
            min_mountain_dist = (BOARD_WIDTH + BOARD_HEIGHT)
        features.append(min_mountain_dist / (BOARD_WIDTH + BOARD_HEIGHT))
        features.append(self.turn / MAX_TURNS)
        features.append(1.0 if self.is_aircraft_surrounded(self.aircraft_pos) else 0.0)
        return features

    def compute_value(self, features):
        return sum(w * x for w, x in zip(self.W, features))

    def drone_compute_value(self, features):
        return sum(w * x for w, x in zip(self.drone_W, features))

    def choose_action(self):
        """
        انتخاب عمل به‌صورت اتوماتیک برای آموزش هواپیما.
        ابتدا جهت‌های ممکن ("up"، "down"، "left"، "right") و شرایط شلیک بررسی می‌شود.
        """
        actions = ["up", "down", "left", "right"]
        best_action = None
        best_value = -float("inf")
        best_features = None

        # بررسی حرکات ساده
        for action in actions:
            new_pos = self._move_toward(self.aircraft_pos, action)
            if new_pos in self.mountains:
                value = -1000  # مجازات برخورد با کوه
                features = self.extract_features()
            else:
                original_pos = self.aircraft_pos
                self.aircraft_pos = new_pos
                features = self.extract_features()
                value = self.compute_value(features)
                self.aircraft_pos = original_pos
            if value > best_value:
                best_value = value
                best_action = action
                best_features = features

        # بررسی امکان شلیک به پهباد (اگر فاصله <= 2 و شرایط برآورده شود)
        for idx, d in enumerate(self.drones):
            if (d is not None and self.aircraft_pos is not None and 
                self.distance(self.aircraft_pos, d) <= 2 and 
                self.aircraft_rockets > 0 and
                self.is_path_clear(self.aircraft_pos, d)):
                shoot_value = 250  # ارزش پیش‌فرض شلیک
                if shoot_value > best_value:
                    best_value = shoot_value
                    best_action = f"shoot_{idx}"
                    best_features = self.extract_features()

        if best_action is None:
            best_action = random.choice(actions)
            best_features = self.extract_features()

        self.training_examples.append((best_features, best_value))
        return best_action, best_value, best_features

    def move_drone(self, dpos, aircraft_pos, occupied):
        if dpos is None:
            return None
        drone_index = self.drones.index(dpos)
        self.previous_drone_positions[drone_index] = dpos
        current_features = self.drone_extract_features(dpos)
        current_value = self.drone_compute_value(current_features)
        directions = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0),
            "stay": (0, 0)
        }
        valid_moves = []
        for (dx, dy) in directions.values():
            new_x = dpos[0] + dx
            new_y = dpos[1] + dy
            if (0 <= new_x < BOARD_WIDTH and 0 <= new_y < BOARD_HEIGHT and 
                (new_x, new_y) not in self.mountains and (new_x, new_y) not in occupied):
                valid_moves.append((new_x, new_y))
        if not valid_moves:
            return dpos
        best_move = None
        best_value = -float('inf')
        for new_pos in valid_moves:
            next_features = self.drone_extract_features(new_pos)
            next_value = self.drone_compute_value(next_features)
            reward = 0
            if self.aircraft_pos is not None:
                current_dist = self.distance(dpos, self.aircraft_pos)
                new_dist = self.distance(new_pos, self.aircraft_pos)
                reward = (current_dist - new_dist) * 10
            total_value = next_value + reward
            if total_value > best_value:
                best_value = total_value
                best_move = new_pos
        if best_move is not None:
            self.drone_training_examples.append((current_features, best_value))
        return best_move if best_move else dpos

    def _move_toward(self, pos, direction):
        x, y = pos
        # اصلاح جهت‌ها: "up" یعنی حرکت به بالا (y کاهش یابد)
        if direction == "up" and y > 0:
            return (x, y - 1)
        elif direction == "down" and y < BOARD_HEIGHT - 1:
            return (x, y + 1)
        elif direction == "left" and x > 0:
            return (x - 1, y)
        elif direction == "right" and x < BOARD_WIDTH - 1:
            return (x + 1, y)
        return pos

    def step(self, manual_action=None):
        """
        در نوبت بازی نهایی، اگر عمل دستی وارد نشده باشد، هیچ اقدامی صورت نمی‌گیرد.
        در غیر این صورت:
          1- پهبادها به صورت خودکار حرکت می‌کنند.
          2- بر اساس عمل دریافتی (manual_action)، هواپیما حرکت می‌کند یا به شلیک اقدام می‌کند.
          3- قوانین برخورد، Reload، Goal و ... اعمال می‌شود.
        """
        if manual_action is None:
            return

        self.turn += 1

        # حرکت پهبادها به صورت خودکار
        occupied = set()
        new_drones = []
        for i, dpos in enumerate(self.drones):
            if dpos is None:
                new_drones.append(None)
                continue
            self.previous_drone_positions[i] = dpos
            new_pos = self.move_drone(dpos, self.aircraft_pos, occupied)
            if new_pos in occupied:
                new_pos = dpos
            new_drones.append(new_pos)
            occupied.add(new_pos)
        for old_pos in self.drones:
            if old_pos is not None:
                self._clear_cell(old_pos)
        self.drones = new_drones
        for dpos in self.drones:
            if dpos is not None:
                self._place_item(dpos, DRONE)

        # پردازش عمل دستی دریافتی:
        self._clear_cell(self.aircraft_pos)
        reward = 0
        terminal = False

        # اگر عمل "space" (فشردن دکمه فاصله) باشد -> تلاش برای شلیک به پهباد نزدیک
        if manual_action == "space":
            shot = False
            for idx, d in enumerate(self.drones):
                if (d is not None and self.aircraft_pos is not None and 
                    self.distance(self.aircraft_pos, d) <= 2 and 
                    self.aircraft_rockets > 0 and 
                    self.is_path_clear(self.aircraft_pos, d)):
                    reward += 250
                    self._clear_cell(d)
                    self.drones[idx] = None
                    self.aircraft_rockets -= 1
                    self.drone_status[idx] = False
                    self.drone_scores[idx] = 0
                    shot = True
                    break
            if not shot:
                # اگر هیچ پهبادی برای شلیک پیدا نشد، هیچ اقدامی انجام نمی‌شود
                pass
        elif manual_action.startswith("shoot"):
            target_idx = int(manual_action.split("_")[1])
            if (self.drones[target_idx] is not None and 
                self.aircraft_pos is not None and 
                self.distance(self.aircraft_pos, self.drones[target_idx]) <= 2 and 
                self.aircraft_rockets > 0 and
                self.is_path_clear(self.aircraft_pos, self.drones[target_idx])):
                reward += 250
                self._clear_cell(self.drones[target_idx])
                self.drones[target_idx] = None
                self.aircraft_rockets -= 1
                self.drone_status[target_idx] = False
                self.drone_scores[target_idx] = 0
        else:
            new_pos = self._move_toward(self.aircraft_pos, manual_action)
            if new_pos in self.mountains:
                reward -= 500
                self.aircraft_pos = self._get_random_empty_cell()
            else:
                self.aircraft_pos = new_pos
                if self.aircraft_pos == self.reload_zone:
                    if self.aircraft_rockets < 3:
                        self.aircraft_rockets += 1
                    self._clear_cell(self.reload_zone)
                    self.reload_zone = self._get_random_empty_cell()
                    self._place_item(self.reload_zone, RELOAD)

        collision_occurred = False
        for i, dpos in enumerate(self.drones):
            if dpos is not None and dpos == self.aircraft_pos:
                collision_occurred = True
                self.drones[i] = None
                self.drone_status[i] = False
                self.drone_scores[i] = 0
                self._clear_cell(dpos)
        if collision_occurred:
            reward -= 1000
            terminal = True
            self.winner = "drones"
            self.loss_reason = "collision with drone"
            self.aircraft_pos = None
        if self.aircraft_pos is not None:
            self._place_item(self.aircraft_pos, AIRCRAFT)
        self.score += reward

        active_drones = [d for d in self.drones if d is not None]
        if len(active_drones) >= 2 and self.aircraft_pos is not None:
            if self.is_aircraft_surrounded(self.aircraft_pos):
                for i in range(NUM_DRONES):
                    if self.drone_status[i] and self.drones[i] is not None and self.distance(self.drones[i], self.aircraft_pos) <= 1:
                        self.drone_scores[i] = 500
                    else:
                        self.drone_scores[i] = 0
                terminal = True
                self.winner = "drones"
                self.loss_reason = "surrounded"
        if len(active_drones) == 0 and self.aircraft_pos == self.goal_pos:
            self.score += 1000
            terminal = True
            self.winner = "aircraft"
            for i in range(NUM_DRONES):
                self.drone_scores[i] = 0
        if self.turn >= MAX_TURNS:
            terminal = True
            if self.winner is None:
                self.winner = "draw"
            else:
                for i in range(NUM_DRONES):
                    self.drone_scores[i] = 0
        if terminal:
            self.game_over = True

    def draw_tk(self, canvas, info_label):
        canvas.delete("all")
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                x1 = x * CELL_SIZE
                y1 = y * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill="white")
        for m in self.mountains:
            mx, my = m
            x1 = mx * CELL_SIZE + 5
            y1 = my * CELL_SIZE + 5
            x2 = (mx+1) * CELL_SIZE - 5
            y2 = (my+1) * CELL_SIZE - 5
            canvas.create_oval(x1, y1, x2, y2, fill="saddlebrown")
            canvas.create_text((mx+0.5)*CELL_SIZE, (my+0.5)*CELL_SIZE, text=MOUNTAIN, fill="white", font=("Arial", 10, "bold"))
        rx, ry = self.reload_zone
        x1 = rx * CELL_SIZE + 5
        y1 = ry * CELL_SIZE + 5
        x2 = (rx+1) * CELL_SIZE - 5
        y2 = (ry+1) * CELL_SIZE - 5
        canvas.create_oval(x1, y1, x2, y2, fill="gold")
        canvas.create_text((rx+0.5)*CELL_SIZE, (ry+0.5)*CELL_SIZE, text=RELOAD, fill="black", font=("Arial", 10, "bold"))
        gx, gy = self.goal_pos
        x1 = gx * CELL_SIZE + 5
        y1 = gy * CELL_SIZE + 5
        x2 = (gx+1) * CELL_SIZE - 5
        y2 = (gy+1) * CELL_SIZE - 5
        canvas.create_oval(x1, y1, x2, y2, fill="limegreen")
        canvas.create_text((gx+0.5)*CELL_SIZE, (gy+0.5)*CELL_SIZE, text=GOAL, fill="white", font=("Arial", 10, "bold"))
        for i, d in enumerate(self.drones):
            if d is not None:
                dx, dy = d
                x1 = dx * CELL_SIZE + 5
                y1 = dy * CELL_SIZE + 5
                x2 = (dx+1) * CELL_SIZE - 5
                y2 = (dy+1) * CELL_SIZE - 5
                canvas.create_oval(x1, y1, x2, y2, fill="crimson")
                canvas.create_text((dx+0.5)*CELL_SIZE, (dy+0.5)*CELL_SIZE, text=f"S{i+1}", fill="white", font=("Arial", 10, "bold"))
        if self.aircraft_pos is not None:
            ax, ay = self.aircraft_pos
            x1 = ax * CELL_SIZE + 5
            y1 = ay * CELL_SIZE + 5
            x2 = (ax+1) * CELL_SIZE - 5
            y2 = (ay+1) * CELL_SIZE - 5
            canvas.create_oval(x1, y1, x2, y2, fill="dodgerblue")
            canvas.create_text((ax+0.5)*CELL_SIZE, (ay+0.5)*CELL_SIZE, text=AIRCRAFT, fill="white", font=("Arial", 10, "bold"))
        info_text = f"Turn: {self.turn}/{MAX_TURNS}\nAircraft Rockets: {self.aircraft_rockets}/3\nAircraft Score: {self.score}\n"
        for i in range(NUM_DRONES):
            status = "Active" if self.drone_status[i] and self.drones[i] is not None else "Destroyed"
            info_text += f"Drone S{i+1}: {self.drone_scores[i]} ({status})\n"
        if self.game_over:
            if self.winner == "drones":
                if self.loss_reason == "collision with drone":
                    info_text += "\nDrones win! (Aircraft collided with a drone)"
                elif self.loss_reason == "surrounded":
                    info_text += "\nDrones win! (Aircraft surrounded)"
                else:
                    info_text += "\nDrones win! (Aircraft destroyed)"
            elif self.winner == "aircraft":
                info_text += "\nAircraft wins! (Mission accomplished)"
            elif self.winner == "draw":
                info_text += "\nGame ended in a draw! (Scores equalized)"
        info_label.config(text=info_text)

# متغیر global جهت ذخیره آخرین عمل دستی کاربر
last_manual_move = None

def on_key_press(event):
    global last_manual_move
    # ثبت کلیدهای جهت و نیز کلید فاصله
    if event.keysym in ["Up", "Down", "Left", "Right"]:
        key_to_action = {
            "Up": "up",
            "Down": "down",
            "Left": "left",
            "Right": "right"
        }
        last_manual_move = key_to_action[event.keysym]
    elif event.keysym.lower() == "space":
        last_manual_move = "space"

def run_final_game(weights, drone_weights):
    game = Game(graphical=True)
    game.W = weights
    game.drone_W = drone_weights

    root = tk.Tk()
    root.title("Air Combat Survival - Final Game (Manual Aircraft Control)")
    canvas_width = BOARD_WIDTH * CELL_SIZE
    canvas_height = BOARD_HEIGHT * CELL_SIZE
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack(side=tk.LEFT)
    info_label = tk.Label(root, text="", justify=tk.LEFT, font=("Arial", 10))
    info_label.pack(side=tk.RIGHT, padx=10, pady=10)

    root.bind("<KeyPress>", on_key_press)

    def game_loop():
        global last_manual_move
        if game.game_over:
            game.draw_tk(canvas, info_label)
            return
        if last_manual_move is None:
            game.draw_tk(canvas, info_label)
            root.after(100, game_loop)
        else:
            action = last_manual_move
            last_manual_move = None
            game.step(manual_action=action)
            game.draw_tk(canvas, info_label)
            root.after(500, game_loop)

    game_loop()
    root.mainloop()

def train_episodes(num_episodes=3000):
    final_weights = None
    final_drone_weights = None
    aircraft_scores = []
    drone_scores = []
    for episode in range(1, num_episodes + 1):
        game = Game(graphical=False)
        while not game.game_over:
            auto_action, _, _ = game.choose_action()
            game.step(manual_action=auto_action)
        for features, target in game.drone_training_examples:
            prediction = game.drone_compute_value(features)
            error = target - prediction
            for i in range(len(game.drone_W)):
                game.drone_W[i] += ALPHA_DRONE * error * features[i]
        final_weights = game.W
        final_drone_weights = game.drone_W
        if game.winner == "aircraft":
            aircraft_scores.append(game.score)
            drone_scores.append(0)
        elif game.winner == "drones":
            drone_scores.append(sum(game.drone_scores))
            aircraft_scores.append(game.score)
        elif game.winner == "draw":
            aircraft_scores.append(game.score)
            drone_scores.append(game.score)
        else:
            aircraft_scores.append(game.score)
            drone_scores.append(0)
        if episode % 500 == 0 or episode == num_episodes:
            print(f"Episode {episode}: Turns = {game.turn}, Winner = {game.winner}, Score = {game.score}")
            print(f"Drone weights: {[round(w, 2) for w in game.drone_W]}")
    with open('trained_weights.pkl', 'wb') as f:
        pickle.dump({
            'aircraft_weights': final_weights,
            'drone_weights': final_drone_weights
        }, f)
    print("Training completed.")
    return final_weights, final_drone_weights

def main():
    print("Starting training (3000 episodes)...")
    final_weights, final_drone_weights = train_episodes(3000)
    print("\nTraining complete. Final aircraft weights:")
    for name, w in zip(aircraft_feature_names, final_weights):
        print(f"{name}: {w:.2f}")
    print("\nFinal drone weights:")
    for name, w in zip(drone_feature_names, final_drone_weights):
        print(f"{name}: {w:.2f}")
    print("\nRunning final game with manual aircraft control...")
    run_final_game(final_weights, final_drone_weights)

if __name__ == "__main__":
    main()
