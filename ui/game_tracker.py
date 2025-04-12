class RotationPoolTracker:
    def __init__(self, player_names):
        """Initialize the game with player names"""
        self.players = []
        self.players_data = {}
        
        # Preserve player order by using an ordered list
        for name in player_names:
            self.players.append(name)
            self.players_data[name] = {
                'score': 0,
                'fouls': 0,
                'status': 'WAITING',
                'pocketed_balls': []  # Add tracking for pocketed balls
            }
        
        # Set the first player as active
        self.current_player_index = 0
        self.players_data[self.players[0]]['status'] = 'ACTIVE'
        
        # Set the next player as "NEXT"
        if len(self.players) > 1:
            self.players_data[self.players[1]]['status'] = 'NEXT'
        
        # Initialize game state
        self.balls_on_table = list(range(1, 16))  # Balls 1-15
        self._lowest_ball = min(self.balls_on_table) if self.balls_on_table else None
        self.last_action = "Game started. " + self.players[0] + " is up first."
        self.game_over = False
        self.cue_ball_on_table = True
        self.last_shot_was_legal = False
        self.last_shot_pocketed_ball = False
        
    @property
    def current_player(self):
        """Get the name of the current player."""
        return self.players[self.current_player_index]
    
    @property
    def active_player(self):
        """Get the name of the active player (who should shoot next)."""
        return self.players[self.current_player_index]
    
    @property
    def lowest_ball(self):
        """Get the lowest numbered ball on the table."""
        return min(self.balls_on_table) if self.balls_on_table else None
    
    def process_shot(self, pocketed_balls, first_ball_hit=None, cue_ball_pocketed=False):
        """Process a shot based on the pocketed balls."""
        # Check if the shot was legal
        legal_shot = False
        if first_ball_hit is not None and first_ball_hit == self.lowest_ball and not cue_ball_pocketed:
            legal_shot = True
            
        # Update game state
        self.last_shot_was_legal = legal_shot
        self.last_shot_pocketed_ball = bool(pocketed_balls)
        
        # Handle pocketed balls based on shot legality
        if legal_shot:
            # Remove legally pocketed balls from the table
            for ball in pocketed_balls:
                if ball in self.balls_on_table:
                    self.balls_on_table.remove(ball)
                    # Track which balls the player pocketed
                    self.players_data[self.current_player]['pocketed_balls'].append(ball)
            
            # Update lowest ball after removing pocketed balls
            self._lowest_ball = min(self.balls_on_table) if self.balls_on_table else None
            
            # Reset fouls for the current player since they made a legal shot
            self.players_data[self.current_player]['fouls'] = 0
            
            # Add points for legally pocketed balls
            if pocketed_balls:
                for ball in pocketed_balls:
                    self.players_data[self.current_player]['score'] += ball
                
                # Player continues their turn if they pocketed a ball
                self.last_action = f"{self.current_player} made a legal shot and pocketed {pocketed_balls}"
            else:
                # No ball pocketed, turn passes to next player
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                self.players_data[self.current_player]['status'] = 'ACTIVE'
                
                # Set the next player as "NEXT"
                next_player_index = (self.current_player_index + 1) % len(self.players)
                for i, player in enumerate(self.players):
                    if i == self.current_player_index:
                        self.players_data[player]['status'] = 'ACTIVE'
                    elif i == next_player_index:
                        self.players_data[player]['status'] = 'NEXT'
                    else:
                        self.players_data[player]['status'] = 'WAITING'
                
                self.last_action = f"{self.players[(self.current_player_index - 1) % len(self.players)]} made a legal shot but didn't pocket any balls"
        else:
            # Illegal shot - balls remain on the table (or are placed back)
            # No points are awarded
            
            # Record a foul
            self.players_data[self.current_player]['fouls'] += 1
            
            # Check for three consecutive fouls
            if self.players_data[self.current_player]['fouls'] >= 3:
                self.players_data[self.current_player]['score'] -= 10  # Penalty
                self.players_data[self.current_player]['fouls'] = 0
                self.last_action = f"{self.current_player} committed 3 consecutive fouls and received a -10 point penalty"
            else:
                self.last_action = f"{self.current_player} committed a foul"
            
            # Turn passes to next player (unless roll-over is invoked)
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.players_data[self.players[self.current_player_index]]['status'] = 'ACTIVE'
            self.players_data[self.players[self.current_player_index]]['fouls'] += 1
        
        # Update cue ball state
        self.cue_ball_on_table = not cue_ball_pocketed
        
        # Check if game is over
        if not self.balls_on_table:
            self.game_over = True
            winner = max(self.players_data.items(), key=lambda x: x[1]['score'])
            self.last_action = f"Game over! {winner[0]} wins with {winner[1]['score']} points"
        
        return {
            "legal_shot": legal_shot,
            "pocketed_balls": pocketed_balls,
            "cue_ball_pocketed": cue_ball_pocketed,
            "next_player": self.active_player,
            "game_over": self.game_over,
            "last_action": self.last_action
        }
    
    def apply_foul(self):
        """Apply a foul to the current player."""
        self.players_data[self.current_player]['fouls'] += 1
        
        # Check for three consecutive fouls
        if self.players_data[self.current_player]['fouls'] >= 3:
            self.players_data[self.current_player]['score'] -= 10  # Penalty
            self.players_data[self.current_player]['fouls'] = 0
            self.last_action = f"{self.current_player} committed 3 consecutive fouls and received a -10 point penalty"
        else:
            self.last_action = f"Foul applied to {self.current_player}"
        
        # Update the current player's status to WAITING
        self.players_data[self.current_player]['status'] = 'WAITING'
        
        # Turn passes to next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        
        # Update player statuses
        self.players_data[self.current_player]['status'] = 'ACTIVE'
        
        # Set the player after next as "NEXT"
        next_player_index = (self.current_player_index + 1) % len(self.players)
        if next_player_index != self.current_player_index:  # Only if there's more than one player
            self.players_data[self.players[next_player_index]]['status'] = 'NEXT'
        
        return {
            "player": self.current_player,
            "fouls": self.players_data[self.current_player]['fouls'],
            "last_action": self.last_action
        }
    
    def apply_rollover(self):
        """Apply the roll-over rule, forcing the previous player to shoot again."""
        # Get the previous player index (the one who just committed a foul)
        prev_player_index = (self.current_player_index - 1) % len(self.players)
        prev_player = self.players[prev_player_index]
        
        # Update the current player's status to WAITING
        self.players_data[self.current_player]['status'] = 'WAITING'
        
        # Force the previous player to shoot again
        self.current_player_index = prev_player_index
        
        # Update player statuses
        self.players_data[prev_player]['status'] = 'ACTIVE'
        
        # Set the player after the previous player as "NEXT"
        next_player_index = (self.current_player_index + 1) % len(self.players)
        if next_player_index != self.current_player_index:  # Only if there's more than one player
            self.players_data[self.players[next_player_index]]['status'] = 'NEXT'
        
        self.last_action = f"Roll-over applied. {prev_player} will shoot again"
        
        return {
            "player": prev_player,
            "last_action": self.last_action
        }
    
    def skip_to_next_player(self):
        """Skip to the next player's turn."""
        # Update the current player's status to WAITING
        self.players_data[self.current_player]['status'] = 'WAITING'
        
        # Move to the next player
        next_player_index = (self.current_player_index + 1) % len(self.players)
        next_player = self.players[next_player_index]
        self.current_player_index = next_player_index
        
        # Update player statuses
        self.players_data[next_player]['status'] = 'ACTIVE'
        
        # Set the player after next as "NEXT"
        next_next_index = (self.current_player_index + 1) % len(self.players)
        if next_next_index != self.current_player_index:  # Only if there's more than one player
            self.players_data[self.players[next_next_index]]['status'] = 'NEXT'
        
        self.last_action = f"Skipped to next player: {next_player}"
        
        return {
            "player": next_player,
            "last_action": self.last_action
        }
    
    def get_game_state(self):
        """Return the current game state as a dictionary."""
        next_player_index = (self.current_player_index + 1) % len(self.players)
        
        # Prepare player data
        players_data = {}
        for i, player in enumerate(self.players):
            status = "WAITING"
            if i == self.current_player_index:
                status = "ACTIVE"
            elif i == next_player_index:
                status = "NEXT"
                
            players_data[player] = {
                'score': self.players_data[player]['score'],
                'fouls': self.players_data[player]['fouls'],
                'status': status,
                'order': i,  # Add player order
                'pocketed_balls': sorted(self.players_data[player]['pocketed_balls'])  # Include pocketed balls
            }
        
        return {
            "current_player": self.current_player,
            "active_player": self.active_player,
            "players_data": players_data,
            "player_order": self.players,  # Include the original player order
            "balls_on_table": sorted(self.balls_on_table),
            "lowest_ball": self.lowest_ball,
            "cue_ball_on_table": self.cue_ball_on_table,
            "game_over": self.game_over,
            "last_action": self.last_action
        }
    
    def __str__(self):
        """String representation of the game state."""
        state = self.get_game_state()
        output = "=== ROTATION POOL GAME STATE ===\n"
        output += f"Current Player: {state['current_player']}\n"
        output += f"Active Player (next to shoot): {state['active_player']}\n"
        output += "\nScores:\n"
        
        for player, score in state['players_data'].items():
            output += f"  {player}: {score['score']} points (Consecutive Fouls: {score['fouls']})\n"
        
        output += f"\nBalls on Table: {state['balls_on_table']}\n"
        output += f"Lowest Ball: {state['lowest_ball']}\n"
        output += f"Cue Ball on Table: {'Yes' if state['cue_ball_on_table'] else 'No'}\n"
        
        if state['game_over']:
            output += "\nGAME OVER!\n"
            winner = max(state['players_data'].items(), key=lambda x: x[1]['score'])
            output += f"Winner: {winner[0]} with {winner[1]['score']} points\n"
            
        return output 