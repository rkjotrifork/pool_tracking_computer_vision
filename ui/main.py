from game_tracker import RotationPoolTracker
from pool_ui import PoolUI
from colorama import Fore

def main():
    # Initialize UI
    ui = PoolUI()
    
    # Display welcome screen
    ui.display_welcome_screen()
    
    # Get player names
    player_names = ui.get_player_names()
    
    # Initialize game tracker
    game = RotationPoolTracker(player_names)
    
    # Main game loop
    while not game.game_over:
        ui.clear_screen()
        
        # Prepare player data for scoreboard
        players_data = {}
        for player in game.player_names:
            status = "WAITING"
            if player == game.current_player:
                status = "ACTIVE"
            elif player == game.player_names[(game.current_player_idx + 1) % game.num_players]:
                status = "NEXT"
                
            players_data[player] = {
                'score': game.scores[player],
                'fouls': game.consecutive_fouls[player],
                'status': status
            }
        
        # Display game state (without pool table)
        ui.display_scoreboard(players_data)
        ui.display_game_info(game.lowest_ball, balls_on_table=game.balls_on_table)
        
        # Display menu and get choice
        choice = ui.display_menu()
        
        if choice == "1":
            # Process a shot
            pocketed_balls, first_ball_hit, cue_ball_pocketed = ui.get_shot_details(
                game.current_player, game.lowest_ball, game.balls_on_table
            )
            
            # Process the shot
            result = game.process_shot(pocketed_balls, first_ball_hit, cue_ball_pocketed)
            
            # Display result
            ui.display_shot_result(result)
            
        elif choice == "2":
            # Apply a foul
            current_player = game.current_player  # Store the current player before applying foul
            result = game.apply_foul()
            ui.display_foul_applied(current_player)  # Display foul for the player who committed it
            
        elif choice == "3":
            # Apply roll-over
            result = game.apply_rollover()
            ui.display_rollover_applied(result["player"])
            
        elif choice == "4":
            # Skip to next player
            next_player_idx = (game.current_player_idx + 1) % game.num_players
            next_player = game.player_names[next_player_idx]
            game.skip_to_player(next_player)
            ui.display_skip_applied(next_player)
            
        elif choice == "5":
            # Quit game
            if input(f"{Fore.RED}Are you sure you want to quit? (y/n): ").lower() == 'y':
                print(f"{Fore.RED}Quitting game...")
                break
        
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.")
            ui.clear_screen()
    
    # Game over
    if game.game_over:
        ui.clear_screen()
        
        # Prepare player data for final scoreboard
        players_data = {}
        winner = max(game.scores.items(), key=lambda x: x[1])
        
        for player in game.player_names:
            status = "WINNER" if player == winner[0] else "FINISHED"
            players_data[player] = {
                'score': game.scores[player],
                'fouls': game.consecutive_fouls[player],
                'status': status
            }
        
        # Display final game state (without pool table)
        ui.display_scoreboard(players_data)
        ui.display_game_info(None, True, winner)
        print(f"{Fore.GREEN}Thanks for playing!")

if __name__ == "__main__":
    main() 