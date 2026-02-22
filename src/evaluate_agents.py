import pickle
import matplotlib.pyplot as plt
import numpy as np
from PartA import Game 
from PartA import MAX_TURNS

def run_part_b(num_games=100):
    """
    Run Part B of the assignment:
    - Plays 100 games using trained weights from Part A
    - Shows graphics only for the first game
    - Records and displays final statistics
    - Plots convergence and results
    """
    
    # Load trained weights from Part A
    try:
        with open('trained_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            final_weights = weights['aircraft_weights']
            final_drone_weights = weights['drone_weights']
    except FileNotFoundError:
        raise Exception("Trained weights file not found. Please run Part A first.")

    # Initialize statistics
    results = {
        'aircraft_wins': 0,
        'drone_wins': 0,
        'draws': 0,
        'best_aircraft_score': -float('inf'),
        'best_drone_score': -float('inf'),
        'scores': []
    }

    # Enable interactive mode for plots
    plt.ion()

    # Run all games
    for game_num in range(1, num_games + 1):
        # Create game instance (only show graphics for first game)
        game = Game(graphical=(game_num == 1))
        
        # Set the trained weights (no further training)
        game.W = final_weights.copy()
        game.drone_W = final_drone_weights.copy()

        # Initialize figure for first game
        if game_num == 1:
            fig = plt.figure(figsize=(10, 6))

        # Run the game to completion
        while not game.game_over and game.turn < MAX_TURNS:
            game.step()
            
            # Draw only for first game
            if game_num == 1:
                game.draw(game.turn, fig)
                plt.pause(0.3)

        # Record results
        if game.winner == "aircraft":
            results['aircraft_wins'] += 1
        elif game.winner == "drones":
            results['drone_wins'] += 1
        else:
            results['draws'] += 1

        # Update best scores
        results['best_aircraft_score'] = max(results['best_aircraft_score'], game.score)
        current_drone_score = sum(game.drone_scores)
        results['best_drone_score'] = max(results['best_drone_score'], current_drone_score)
        
        # Store scores for plotting
        results['scores'].append({
            'game': game_num,
            'aircraft_score': game.score,
            'drone_score': current_drone_score,
            'winner': game.winner
        })

        # Print progress every 10 games
        if game_num % 10 == 0 or game_num == num_games:
            print(f"Game {game_num}: Winner = {game.winner}, Aircraft = {game.score}, Drones = {current_drone_score}")

    # Disable interactive mode
    plt.ioff()

    # Calculate percentages
    total_games = results['aircraft_wins'] + results['drone_wins'] + results['draws']
    results['aircraft_win_percent'] = (results['aircraft_wins'] / total_games) * 100
    results['drone_win_percent'] = (results['drone_wins'] / total_games) * 100
    results['draw_percent'] = (results['draws'] / total_games) * 100

    # Print final results
    print("\n=== Final Results ===")
    print(f"Total Games: {total_games}")
    print(f"Aircraft Wins: {results['aircraft_wins']} ({results['aircraft_win_percent']:.1f}%)")
    print(f"Drone Wins: {results['drone_wins']} ({results['drone_win_percent']:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draw_percent']:.1f}%)")
    print(f"\nBest Aircraft Score: {results['best_aircraft_score']}")
    print(f"Best Drone Score: {results['best_drone_score']}")

    # Plot results
    plot_results(results)

def plot_results(results):
    """Plot the convergence and final results"""
    
    # Prepare data for plotting
    games = [r['game'] for r in results['scores']]
    aircraft_scores = [r['aircraft_score'] for r in results['scores']]
    drone_scores = [r['drone_score'] for r in results['scores']]
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Score trends
    plt.subplot(1, 3, 1)
    plt.plot(games, aircraft_scores, label='Aircraft')
    plt.plot(games, drone_scores, label='Drones')
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title('Score Trends Over Games')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Win distribution
    plt.subplot(1, 3, 2)
    labels = ['Aircraft', 'Drones', 'Draws']
    sizes = [results['aircraft_win_percent'], results['drone_win_percent'], results['draw_percent']]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Win Distribution')
    
    # Plot 3: Score distribution
    plt.subplot(1, 3, 3)
    bins = np.linspace(min(aircraft_scores + drone_scores), max(aircraft_scores + drone_scores), 20)
    plt.hist(aircraft_scores, bins, alpha=0.5, label='Aircraft')
    plt.hist(drone_scores, bins, alpha=0.5, label='Drones')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting Part B: Running 100 games with trained agents...")
    run_part_b(100)