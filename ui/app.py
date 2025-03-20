from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from game_tracker import RotationPoolTracker
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Global game instance (for simplicity - in production you'd use a database)
game_instance = None

@app.route('/')
def index():
    """Main page - redirects to setup if no game is in progress"""
    global game_instance
    if game_instance is None:
        return redirect(url_for('setup'))
    return render_template('index.html')

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Player setup page"""
    global game_instance
    
    if request.method == 'POST':
        player_names = []
        
        # Get all player inputs in order
        for i in range(1, 100):  # Support up to 99 players (more than enough)
            key = f'player{i}'
            if key in request.form and request.form[key].strip():
                player_names.append(request.form[key].strip())
        
        if len(player_names) >= 2:  # Need at least 2 players
            game_instance = RotationPoolTracker(player_names)
            return redirect(url_for('index'))
        
    return render_template('setup.html')

@app.route('/api/game_state')
def game_state():
    """API endpoint to get the current game state"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    return jsonify(game_instance.get_game_state())

@app.route('/api/process_shot', methods=['POST'])
def process_shot():
    """API endpoint to process a shot"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    data = request.json
    pocketed_balls = data.get('pocketed_balls', [])
    
    # Assume legal shot (first ball hit is lowest ball, cue ball not pocketed)
    first_ball_hit = game_instance.lowest_ball
    cue_ball_pocketed = False
    
    result = game_instance.process_shot(pocketed_balls, first_ball_hit, cue_ball_pocketed)
    return jsonify(result)

@app.route('/api/apply_foul', methods=['POST'])
def apply_foul():
    """API endpoint to apply a foul"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    result = game_instance.apply_foul()
    return jsonify(result)

@app.route('/api/apply_rollover', methods=['POST'])
def apply_rollover():
    """API endpoint to apply roll-over"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    # Check if the last action contains "foul" to ensure rollover is valid
    if "foul" not in game_instance.last_action.lower():
        return jsonify({
            "error": "Cannot apply roll-over. Previous player did not commit a foul.",
            "last_action": game_instance.last_action
        }), 400
    
    result = game_instance.apply_rollover()
    return jsonify(result)

@app.route('/api/skip_player', methods=['POST'])
def skip_player():
    """API endpoint to skip to next player"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    result = game_instance.skip_to_next_player()
    return jsonify(result)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """API endpoint to start a new game"""
    global game_instance
    game_instance = None
    return jsonify({"status": "success", "message": "New game started"})

@app.route('/api/update_score', methods=['POST'])
def update_score():
    """API endpoint to manually update a player's score"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    data = request.json
    player_name = data.get('player')
    new_score = data.get('score')
    
    if player_name not in game_instance.players_data:
        return jsonify({"error": f"Player {player_name} not found"}), 400
    
    try:
        new_score = int(new_score)
    except (ValueError, TypeError):
        return jsonify({"error": "Score must be a valid number"}), 400
    
    # Update the player's score
    game_instance.players_data[player_name]['score'] = new_score
    game_instance.last_action = f"Score for {player_name} manually updated to {new_score}"
    
    return jsonify(game_instance.get_game_state())

@app.route('/api/update_fouls', methods=['POST'])
def update_fouls():
    """API endpoint to manually update a player's fouls"""
    global game_instance
    if game_instance is None:
        return jsonify({"error": "No game in progress"}), 400
    
    data = request.json
    player_name = data.get('player')
    new_fouls = data.get('fouls')
    
    if player_name not in game_instance.players_data:
        return jsonify({"error": f"Player {player_name} not found"}), 400
    
    try:
        new_fouls = int(new_fouls)
    except (ValueError, TypeError):
        return jsonify({"error": "Fouls must be a valid number"}), 400
    
    # Update the player's fouls
    game_instance.players_data[player_name]['fouls'] = new_fouls
    game_instance.last_action = f"Fouls for {player_name} manually updated to {new_fouls}"
    
    return jsonify(game_instance.get_game_state())

if __name__ == '__main__':
    app.run(debug=True) 