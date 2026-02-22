# Air Combat Survival: Function Approximation Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Course](https://img.shields.io/badge/Course-Machine%20Learning%20(Spring%202025)-green.svg)]()

This project implements **Linear Function Approximation** for multi-agent reinforcement learning within an intense air combat simulation. Based on **Chapter 1 of Tom Mitchell's *Machine Learning***, this assignment tasks an **Aircraft (A)** with eliminating 4 Suicide Drones (S1–S4) and reaching a Goal area, while the drones coordinate to surround and destroy the Aircraft.

The system uses **Temporal Difference (TD) learning** principles with linear value functions to approximate the value of states (
$\\hat{V}(s)$
) for both agents, enabling them to make optimal decisions in a stochastic environment with obstacles, reload zones, and dynamic threats.


## 🎮 Game Mechanics & Rules

The simulation takes place on a **15×10 grid-based airspace**.

| Element | Symbol | Description |
| :--- | :---: | :--- |
| **Aircraft** | `A` | Player agent. Starts with 2 rockets (Max 3). Must eliminate drones and reach Goal. |
| **Suicide Drones** | `S` | 4 AI agents. Coordinate to surround or swarm the Aircraft. Share weights. |
| **Goal** | `G` | Fixed at corner (15, 10). Reached after eliminating all drones for **+1000 pts**. |
| **Reload Zone** | `R` | Randomly placed. Grants **+1 rocket**. Regenerates after use. |
| **Mountains** | `M` | 10 Obstacles. Crashing costs **-500 pts** and respawns Aircraft. |

### Win/Loss Conditions
*   **🏆 Aircraft Wins:** All drones destroyed **AND** Goal reached.
*   **💀 Drones Win:** 
    *   Surround Aircraft (4 sides OR 2 adjacent sides).
    *   2+ drones within 1 block of Aircraft.
    *   Aircraft crashes into a drone.
*   **🤝 Draw:** 30 turns pass without a decisive winner.


## Machine Learning Methodology

### Algorithm
*   **Method:** Linear Value Function Approximation.
*   **Update Rule:** Least Mean Squares (LMS) / Gradient Descent.
*   **Value Functions:**
      *   **Aircraft:**       $$\hat{V}_A(s) = W_A^T x(s)$$
      *   **Drones:**          $$\hat{V}_D(s) = W_D^T x(s)$$

  
 
*   **Weight Update:**
    $$W \leftarrow W + \alpha (V_{\text{train}}(s) - \hat{V}(s)) x(s)$$

### Feature Engineering
The agents evaluate states using normalized feature vectors $x(s)$.

#### ✈️ Aircraft Features (12 Dimensions)
1.  **Bias:** Constant 1.0.
2.  **Distance to Goal:** Normalized Manhattan distance.
3.  **Rockets:** Normalized count (0-3).
4.  **Min Drone Distance:** Normalized distance to nearest drone.
5.  **Drones within 1 Block:** Count (Threat level).
6.  **Drones within 2 Blocks:** Count (Shootable range).
7.  **Distance to Reload:** Normalized distance.
8.  **Min Mountain Distance:** Normalized safety margin.
9.  **Turn Ratio:** Current turn / Max turns.
10. **Avg Drone Distance:** Normalized average distance to all drones.
11. **Need Reload:** Binary (1 if rockets < 1).
12. **All Drones Destroyed:** Binary (1 if all dead).

#### 🛸 Drone Features (9 Dimensions)
1.  **Bias:** Constant 1.0.
2.  **Distance to Aircraft:** Normalized Manhattan distance.
3.  **Distance to Goal:** Normalized (to block aircraft).
4.  **Distance to Nearest Drone:** Normalized (for coordination).
5.  **Aircraft in Sight:** Binary (within range).
6.  **Nearby Drones Count:** Normalized count (for swarming).
7.  **Distance to Mountain:** Normalized safety margin.
8.  **Turn Ratio:** Current turn / Max turns.
9.  **Is Surrounding:** Binary (1 if currently surrounding aircraft).

### Reward Structure
| Event | Aircraft Reward | Drone Reward |
| :--- | :---: | :---: |
| **Destroy Enemy** | +250 (per drone) | -250 (per drone) |
| **Win Game** | +1000 (Goal) | +500 (per drone) |
| **Loss/Crash** | -1000 (Destroyed) | - |
| **Mountain Crash** | -500 | - |
| **Draw** | 0 | 0 |


## Project Structure

```           
air-combat-survival/
│
├── src/                        
│   ├── __init__.py             
│   ├── train_agents.py         # Core training logic, Game class, Matplotlib Visualization (Part A)
│   ├── evaluate_agents.py      # Evaluation script (100 games), Statistics & Plotting (Part B)
│   ├── interactive_gui.py      # Interactive GUI (Tkinter) for Human vs. AI (Part C)
│
├── assets/                     
│   ├── trained_weights.pkl     # Generated file containing trained weights (Aircraft & Drones)
│       
├── README.md                   # Project Documentation
```               


## Results & Visualizations
### Learning Progress (Part A)
A line plot showing the convergence of Aircraft and Drone scores over 3000 episodes. Ideally, the Aircraft score should stabilize or increase as it learns to avoid drones and reach the goal.
### Performance Statistics (Part B)
1. Score Trends: Line plot comparing scores over 100 test games.
2. Win Distribution: Pie chart showing percentages (Aircraft Win vs. Drone Win vs. Draw).
3. Score Distribution: Histogram showing frequency of scores achieved.
### GUI (Part C)
Real-time rendering using tkinter:
+ Blue Circle: Aircraft
+ Red Circles: Drones
+ Brown Circles: Mountains
+ Gold Circle: Reload Zone
+ Green Circle: Goal

## Reference:
Tom Mitchell, Machine Learning, Chapter 1.

