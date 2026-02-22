import random
import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Game settings
BOARD_WIDTH = 15
BOARD_HEIGHT = 10
MAX_TURNS = 40

# Symbols
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

# Aircraft features
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

# Drone features
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
        for dir_name, (dx, dy) in directions.items():
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

    def simulate_action(self, action):
        sim = copy.deepcopy(self)
        sim._clear_cell(sim.aircraft_pos)
        reward = 0
        terminal = False

        if action.startswith("shoot"):
            target_idx = int(action.split("_")[1])
            if (sim.drones[target_idx] is not None and 
                sim.aircraft_pos is not None and 
                sim.distance(sim.aircraft_pos, sim.drones[target_idx]) <= 2 and 
                sim.aircraft_rockets > 0 and
                sim.is_path_clear(sim.aircraft_pos, sim.drones[target_idx])):
                
                reward += 250
                sim._clear_cell(sim.drones[target_idx])
                sim.drones[target_idx] = None
                sim.aircraft_rockets -= 1
        else:
            new_pos = self._move_toward(sim.aircraft_pos, action)
            if new_pos in sim.mountains:
                reward -= 500
                sim.aircraft_pos = sim._get_random_empty_cell()
            else:
                sim.aircraft_pos = new_pos
                if sim.aircraft_pos == sim.reload_zone:
                    if sim.aircraft_rockets < 3:
                        sim.aircraft_rockets += 1
                    sim._clear_cell(sim.reload_zone)
                    sim.reload_zone = sim._get_random_empty_cell()
                    sim._place_item(sim.reload_zone, RELOAD)
        if sim.aircraft_pos is not None:
            sim._place_item(sim.aircraft_pos, AIRCRAFT)
        
        occupied = set()
        new_drones = []
        for dpos in sim.drones:
            if dpos is None:
                new_drones.append(None)
                continue
            new_pos = self.move_drone(dpos, sim.aircraft_pos, occupied)
            if new_pos in occupied:
                new_pos = dpos
            new_drones.append(new_pos)
            occupied.add(new_pos)
        sim.drones = new_drones
        for dpos in sim.drones:
            if dpos is not None:
                sim._place_item(dpos, DRONE)
        sim.turn += 1

        if sim.aircraft_pos is not None:
            for i, dpos in enumerate(sim.drones):
                if dpos is not None and dpos == sim.aircraft_pos:
                    reward -= 1000
                    terminal = True
                    sim.winner = "drones"
                    sim.aircraft_pos = None
                    sim._clear_cell(dpos)
                    sim.drones[i] = None

        if all(d is None for d in sim.drones) and sim.aircraft_pos == sim.goal_pos:
            reward += 1000
            terminal = True
            sim.winner = "aircraft"

        active_drones = [d for d in sim.drones if d is not None]
        if len(active_drones) >= 2 and sim.aircraft_pos is not None and sim.is_aircraft_surrounded(sim.aircraft_pos):
            adjacent_drones = [pos for pos in [
                (sim.aircraft_pos[0], sim.aircraft_pos[1]+1),
                (sim.aircraft_pos[0], sim.aircraft_pos[1]-1),
                (sim.aircraft_pos[0]-1, sim.aircraft_pos[1]),
                (sim.aircraft_pos[0]+1, sim.aircraft_pos[1])
            ] if pos in [d for d in sim.drones if d is not None]]
            
            reward -= 1000
            for _ in range(len(adjacent_drones)):
                reward += 500
            terminal = True
            sim.winner = "drones"

        if sim.turn >= MAX_TURNS:
            terminal = True
            # Make scores equal when turn 30 is reached
            if sim.winner is None:
                sim.winner = "draw"
                if sim.score > 0:
                    drone_total = sum(500 for d in sim.drones if d is not None)
                    sim.score = drone_total = (sim.score + drone_total) // 2
                    for i in range(len(sim.drones)):
                        if sim.drones[i] is not None:
                            sim.drone_scores[i] = drone_total // len([d for d in sim.drones if d is not None])

        features_next = sim.extract_features()
        return reward, terminal, features_next

    def choose_action(self):
        close_drones = [(i, d) for i, d in enumerate(self.drones) 
                       if d is not None 
                       and self.distance(self.aircraft_pos, d) <= 2
                       and self.is_path_clear(self.aircraft_pos, d)]
        
        if close_drones and self.aircraft_rockets > 0:
            i, _ = close_drones[0]
            return f"shoot_{i}", self.extract_features(), self.compute_value(self.extract_features())
        
        all_drones_destroyed = all(d is None for d in self.drones)
        
        if all_drones_destroyed:
            dx = self.goal_pos[0] - self.aircraft_pos[0]
            dy = self.goal_pos[1] - self.aircraft_pos[1]
            
            if abs(dx) > abs(dy):
                action = "right" if dx > 0 else "left"
            else:
                action = "up" if dy > 0 else "down"
                
            if action in self.possible_actions():
                return action, self.extract_features(), self.compute_value(self.extract_features())
        
        if self.aircraft_rockets == 0 and not all_drones_destroyed:
            dx = self.reload_zone[0] - self.aircraft_pos[0]
            dy = self.reload_zone[1] - self.aircraft_pos[1]
            
            if abs(dx) > abs(dy):
                action = "right" if dx > 0 else "left"
            else:
                action = "up" if dy > 0 else "down"
                
            if action in self.possible_actions():
                return action, self.extract_features(), self.compute_value(self.extract_features())
        
        if not all_drones_destroyed:
            active_drones = [d for d in self.drones if d is not None]
            if active_drones:
                nearest_drone = min(active_drones, 
                                  key=lambda d: self.distance(self.aircraft_pos, d))
                dx = nearest_drone[0] - self.aircraft_pos[0]
                dy = nearest_drone[1] - self.aircraft_pos[1]
                
                if abs(dx) > abs(dy):
                    action = "right" if dx > 0 else "left"
                else:
                    action = "up" if dy > 0 else "down"
                    
                if action in self.possible_actions():
                    return action, self.extract_features(), self.compute_value(self.extract_features())
        
        possible_actions = self.possible_actions()
        best_action = None
        best_estimated = -float('inf')
        current_features = self.extract_features()
        current_value = self.compute_value(current_features)
        for action in possible_actions:
            r_sim, terminal_sim, features_sim = self.simulate_action(action)
            estimated = r_sim + (0 if terminal_sim else self.compute_value(features_sim))
            if estimated > best_estimated:
                best_estimated = estimated
                best_action = action
        
        self.training_examples.append((current_features, best_estimated))
        
        return best_action if best_action else random.choice(possible_actions), current_features, current_value

    def possible_actions(self):
        actions = ["up", "down", "left", "right"]
        for i, dpos in enumerate(self.drones):
            if (dpos is not None and 
                self.aircraft_pos is not None and 
                self.distance(self.aircraft_pos, dpos) <= 2 and 
                self.aircraft_rockets > 0 and
                self.is_path_clear(self.aircraft_pos, dpos)):
                actions.append(f"shoot_{i}")
        return actions

    def _move_toward(self, pos, direction):
        x, y = pos
        if direction == "up" and y < BOARD_HEIGHT - 1:
            return (x, y + 1)
        elif direction == "down" and y > 0:
            return (x, y - 1)
        elif direction == "left" and x > 0:
            return (x - 1, y)
        elif direction == "right" and x < BOARD_WIDTH - 1:
            return (x + 1, y)
        return pos

    def step(self):
        if self.game_over:
            return
            
        self.turn += 1
        
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
        
        if self.aircraft_pos is None:
            return
            
        action, current_features, current_value = self.choose_action()
        self._clear_cell(self.aircraft_pos)
        reward = 0
        terminal = False

        if action.startswith("shoot"):
            target_idx = int(action.split("_")[1])
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
            new_pos = self._move_toward(self.aircraft_pos, action)
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
                # Make scores equal when turn 30 is reached
                if self.score > 0:
                    drone_total = sum(500 for d in self.drones if d is not None)
                    self.score = drone_total = (self.score + drone_total) // 2
                    for i in range(NUM_DRONES):
                        if self.drones[i] is not None:
                            self.drone_scores[i] = drone_total // len([d for d in self.drones if d is not None])
            else:
                for i in range(NUM_DRONES):
                    self.drone_scores[i] = 0

        if terminal:
            # Update aircraft weights using LMS
            for features, target in self.training_examples:
                prediction = self.compute_value(features)
                error = target - prediction
                for i in range(len(self.W)):
                    self.W[i] += ALPHA * error * features[i]
            
            # Update drone weights using LMS
            for features, target in self.drone_training_examples:
                prediction = self.drone_compute_value(features)
                error = target - prediction
                for i in range(len(self.drone_W)):
                    self.drone_W[i] += ALPHA_DRONE * error * features[i]
            
            self.training_examples = []
            self.drone_training_examples = []
        
        if terminal:
            self.game_over = True

    def draw(self, step_number, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(10, 6))
        else:
            fig.clf()
        
        gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 1])
        
        ax = fig.add_subplot(gs[0])
        ax.set_xlim(-0.5, BOARD_WIDTH - 0.5)
        ax.set_ylim(-0.5, BOARD_HEIGHT - 0.5)
        ax.set_aspect('equal')
        ax.set_title("Air Combat Survival")
        
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                       edgecolor="gray", facecolor="white", 
                                       linewidth=0.5)
                ax.add_patch(rect)
        
        for m in self.mountains:
            mx, my = m
            circ = patches.Circle((mx, my), 0.4, color="saddlebrown")
            ax.add_patch(circ)
            ax.text(mx, my, MOUNTAIN, ha="center", va="center", 
                   fontsize=10, color="white", weight='bold')
        
        rx, ry = self.reload_zone
        circ = patches.Circle((rx, ry), 0.4, color="gold")
        ax.add_patch(circ)
        ax.text(rx, ry, RELOAD, ha="center", va="center", 
               fontsize=10, color="black", weight='bold')
        
        gx, gy = self.goal_pos
        circ = patches.Circle((gx, gy), 0.4, color="limegreen")
        ax.add_patch(circ)
        ax.text(gx, gy, GOAL, ha="center", va="center", 
               fontsize=10, color="white", weight='bold')
        
        for i, d in enumerate(self.drones):
            if d is not None:
                dx, dy = d
                circ = patches.Circle((dx, dy), 0.4, color="crimson")
                ax.add_patch(circ)
                ax.text(dx, dy, f"S{i+1}", ha="center", va="center", 
                       fontsize=10, color="white", weight='bold')
        
        if self.aircraft_pos is not None:
            ax.add_patch(patches.Circle((self.aircraft_pos[0], self.aircraft_pos[1]), 
                        0.4, color="dodgerblue"))
            ax.text(self.aircraft_pos[0], self.aircraft_pos[1], AIRCRAFT, 
                   ha="center", va="center", fontsize=10, color="white", weight='bold')
        
        ax_info = fig.add_subplot(gs[1])
        ax_info.set_facecolor('lightgoldenrodyellow')
        ax_info.axis('off')
        
        ax_info.text(0.5, 0.95, "Game Status", ha='center', va='center', 
                    fontsize=12, weight='bold', color='navy')
        
        info_text = [
            f"Turn: {step_number}/{MAX_TURNS}",
            f"Aircraft Rockets: {self.aircraft_rockets}/3",
            f"Aircraft Score: {self.score}",
            "\nDrone Scores:"
        ]
        
        for i in range(NUM_DRONES):
            status = "Active" if self.drone_status[i] and self.drones[i] is not None else "Destroyed"
            info_text.append(f"Drone S{i+1}: {self.drone_scores[i]} ({status})")
        
        if self.game_over:
            if self.winner == "drones":
                if hasattr(self, 'loss_reason') and self.loss_reason == "collision with drone":
                    info_text.append("\nDrones win! (Aircraft collided with a drone)")
                elif hasattr(self, 'loss_reason') and self.loss_reason == "surrounded":
                    info_text.append("\nDrones win! (Aircraft surrounded)")
                else:
                    info_text.append("\nDrones win! (Aircraft destroyed)")
            elif self.winner == "aircraft":
                info_text.append("\nAircraft wins! (Mission accomplished)")
            elif self.winner == "draw":
                info_text.append("\nGame ended in a draw! (Scores equalized)")
            elif self.winner == "timeout":
                info_text.append("\nGame ended (Time out)")
        
        ax_info.text(0.1, 0.85, "\n".join(info_text), ha='left', va='top', 
                    fontsize=10, color='black', linespacing=1.5)
        
        plt.tight_layout()
        plt.draw()

def train_episodes(num_episodes=3000):
    final_weights = None
    final_drone_weights = None
    aircraft_scores = []
    drone_scores = []
    
    for episode in range(1, num_episodes + 1):
        game = Game(graphical=False)
        while not game.game_over:
            game.step()
        
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
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes+1), aircraft_scores, label="Aircraft Score")
    plt.plot(range(1, num_episodes+1), drone_scores, label="Drone Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Learning Progress")
    plt.legend()
    plt.show()
    
    with open('trained_weights.pkl', 'wb') as f:
        pickle.dump({
            'aircraft_weights': final_weights,
            'drone_weights': final_drone_weights
        }, f)
    
    return final_weights, final_drone_weights

def run_final_game(weights, drone_weights):
    game = Game(graphical=True)
    game.W = weights
    game.drone_W = drone_weights
    print("\nFinal game visualization (step by step):")
    
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    
    for step in range(1, MAX_TURNS + 1):
        if game.game_over:
            game.draw(step, fig)
            plt.pause(1)
            if game.winner == "drones":
                if hasattr(game, 'loss_reason') and game.loss_reason == "collision with drone":
                    print("\nDrones win! (Aircraft collided with a drone)")
                elif hasattr(game, 'loss_reason') and game.loss_reason == "surrounded":
                    print("\nDrones win! (Aircraft surrounded)")
                else:
                    print("\nDrones win! (Aircraft destroyed)")
            elif game.winner == "aircraft":
                print("\nAircraft wins! (All drones destroyed and reached goal)")
            elif game.winner == "draw":
                print("\nGame ended in a draw! (Scores equalized)")
            elif game.winner == "timeout":
                print("\nGame ended after reaching maximum turns (30 turns)")

            print(f"\nFinal scores:")
            print(f"Aircraft: {game.score}")
            print("Drones status and scores:")
            for i in range(NUM_DRONES):
                status = "Active" if game.drone_status[i] else "Destroyed"
                print(f"Drone {i+1}: {status} | Score: {game.drone_scores[i]}")
            break

        game.step()
        game.draw(step, fig)
        plt.pause(0.5)

    if not game.game_over:
        game.draw(MAX_TURNS, fig)
        print("\nGame ended after reaching maximum turns (30 turns)")
        print(f"\nFinal scores:")
        print(f"Aircraft: {game.score}")
        print("Drones status and scores:")
        for i in range(NUM_DRONES):
            status = "Active" if game.drone_status[i] else "Destroyed"
            print(f"Drone {i+1}: {status} | Score: {game.drone_scores[i]}")

    plt.ioff()
    plt.show()

def main():
    print("Starting function approximation training (3000 episodes)...")
    final_weights, final_drone_weights = train_episodes(3000)
    print("\nTraining complete. Final aircraft weights:")
    for name, w in zip(aircraft_feature_names, final_weights):
        print(f"{name}: {w:.2f}")
    print("\nFinal drone weights:")
    for name, w in zip(drone_feature_names, final_drone_weights):
        print(f"{name}: {w:.2f}")
    print("\nRunning final game with visualization:")
    
    run_final_game(final_weights, final_drone_weights)

if __name__ == "__main__":
    main()