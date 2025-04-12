import os
import time
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

class PoolUI:
    def __init__(self):
        self.table_width = 60
        self.table_height = 25
    
    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_pool_table(self, balls_on_table):
        """Draw an ASCII art pool table with the balls on it."""
        # Table border
        top_border = f"{Fore.GREEN}+{'-' * (self.table_width - 2)}+"
        side_border = f"{Fore.GREEN}|{' ' * (self.table_width - 2)}|"
        
        # Draw the table
        print(top_border)
        
        # Draw pockets
        print(f"{Fore.GREEN}O{' ' * (self.table_width - 4)}O")
        
        # Draw the felt
        for _ in range(self.table_height - 6):
            print(side_border)
        
        # Draw balls on the table
        ball_positions = self._calculate_ball_positions(balls_on_table)
        table_with_balls = self._place_balls_on_table(ball_positions)
        for line in table_with_balls:
            print(line)
        
        # Draw bottom pockets
        print(f"{Fore.GREEN}O{' ' * (self.table_width - 4)}O")
        
        # Draw bottom border
        print(top_border)
    
    def _calculate_ball_positions(self, balls_on_table):
        """Calculate positions for balls on the table."""
        positions = {}
        available_positions = []
        
        # Create a grid of available positions
        for row in range(3, self.table_height - 7):
            for col in range(4, self.table_width - 6, 4):
                available_positions.append((row, col))
        
        # Assign positions to balls
        for i, ball in enumerate(sorted(balls_on_table)):
            if i < len(available_positions):
                positions[ball] = available_positions[i]
        
        return positions
    
    def _place_balls_on_table(self, ball_positions):
        """Place balls on the table at their calculated positions."""
        # Create empty table
        table = []
        for _ in range(self.table_height - 4):
            table.append(f"{Fore.GREEN}|{' ' * (self.table_width - 2)}|")
        
        # Place balls on the table
        for ball, (row, col) in ball_positions.items():
            # Convert row to table index
            table_row = row - 3
            
            # Get the current row
            current_row = table[table_row]
            
            # Determine ball color based on number
            if ball <= 8:
                ball_color = Fore.YELLOW
            else:
                ball_color = Fore.RED
                
            # Replace the character at the column position with the ball
            new_row = (current_row[:col] + 
                      f"{ball_color}{ball:2d}" + 
                      current_row[col+2:])
            
            table[table_row] = new_row
            
        return table
    
    def display_scoreboard(self, players_data):
        """Display a stylish scoreboard with player information."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 50}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'ROTATION POOL SCOREBOARD':^50}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 50}")
        
        # Header
        print(f"{Fore.WHITE}{Style.BRIGHT}{'PLAYER':12}{'SCORE':10}{'FOULS':10}{'STATUS':18}")
        print(f"{Fore.CYAN}{'-' * 50}")
        
        # Player data
        for player, data in players_data.items():
            score = data['score']
            fouls = data['fouls']
            status = data['status']
            
            # Determine color based on status
            if status == 'ACTIVE':
                name_color = Fore.GREEN
                status_text = f"{Fore.GREEN}→ ACTIVE"
            elif status == 'NEXT':
                name_color = Fore.YELLOW
                status_text = f"{Fore.YELLOW}NEXT"
            else:
                name_color = Fore.WHITE
                status_text = f"{Fore.WHITE}WAITING"
            
            print(f"{name_color}{player:12}{Fore.WHITE}{score:10}{fouls:10}  {status_text}")
        
        print(f"{Fore.CYAN}{'-' * 50}\n")
    
    def display_game_info(self, lowest_ball, game_over=False, winner=None, balls_on_table=None):
        """Display game information."""
        print(f"{Fore.CYAN}{'=' * 50}")
        
        if game_over:
            print(f"{Fore.MAGENTA}{Style.BRIGHT}{'GAME OVER!':^50}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}{f'WINNER: {winner[0]} with {winner[1]} points':^50}")
        else:
            print(f"{Fore.WHITE}Lowest Ball on Table: {Fore.YELLOW}{lowest_ball}")
            if balls_on_table:
                print(f"{Fore.WHITE}Balls on Table: {Fore.YELLOW}{sorted(balls_on_table)}")
        
        print(f"{Fore.CYAN}{'=' * 50}\n")
    
    def display_menu(self):
        """Display the game menu."""
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'GAME MENU':^50}")
        print(f"{Fore.CYAN}{'=' * 50}")
        print(f"{Fore.WHITE}1. {Fore.GREEN}Process a shot (legal)")
        print(f"{Fore.WHITE}2. {Fore.RED}Apply a foul")
        print(f"{Fore.WHITE}3. {Fore.YELLOW}Apply roll-over")
        print(f"{Fore.WHITE}4. {Fore.BLUE}Skip to next player")
        print(f"{Fore.WHITE}5. {Fore.RED}Quit game")
        print(f"{Fore.CYAN}{'=' * 50}")
        
        choice = input(f"{Fore.WHITE}Enter your choice (1-5): ")
        return choice
    
    def get_shot_details(self, current_player, lowest_ball, balls_on_table):
        """Get shot details from the user."""
        self.clear_screen()
        print(f"\n{Fore.GREEN}{Style.BRIGHT}{'PROCESSING SHOT':^50}")
        print(f"{Fore.GREEN}{'=' * 50}")
        print(f"{Fore.WHITE}Current Player: {Fore.YELLOW}{current_player}")
        print(f"{Fore.WHITE}Lowest Ball: {Fore.YELLOW}{lowest_ball}")
        print(f"{Fore.WHITE}Balls on Table: {Fore.YELLOW}{sorted(balls_on_table)}")
        print(f"{Fore.GREEN}{'=' * 50}\n")
        
        # Get pocketed balls
        pocketed_balls_input = input(f"{Fore.WHITE}Enter balls pocketed (comma-separated numbers, or 0 if none): ")
        if pocketed_balls_input.strip() == "0":
            pocketed_balls = []
        elif pocketed_balls_input.strip():
            try:
                pocketed_balls = [int(b.strip()) for b in pocketed_balls_input.split(",")]
            except ValueError:
                print(f"{Fore.RED}Invalid input! Using empty list.")
                pocketed_balls = []
        else:
            pocketed_balls = []
        
        # Assume legal shot (first ball hit is lowest ball, cue ball not pocketed)
        first_ball_hit = lowest_ball
        cue_ball_pocketed = False
        
        return pocketed_balls, first_ball_hit, cue_ball_pocketed
    
    def display_shot_result(self, result):
        """Display the result of a shot."""
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'SHOT RESULT':^50}")
        print(f"{Fore.MAGENTA}{'=' * 50}")
        
        if result['legal_shot']:
            print(f"{Fore.GREEN}Legal shot: Yes")
            if result['pocketed_balls']:
                print(f"{Fore.YELLOW}Pocketed balls: {result['pocketed_balls']}")
            else:
                print(f"{Fore.WHITE}No balls pocketed")
        else:
            print(f"{Fore.RED}Legal shot: No")
            if result['pocketed_balls']:
                print(f"{Fore.RED}Illegally pocketed balls (placed back on table): {result['pocketed_balls']}")
            else:
                print(f"{Fore.WHITE}No balls pocketed")
        
        if result['cue_ball_pocketed']:
            print(f"{Fore.RED}Cue ball was pocketed!")
        
        print(f"{Fore.WHITE}Next player: {Fore.GREEN}{result['next_player']}")
        print(f"{Fore.MAGENTA}{'=' * 50}")
        
        input(f"\n{Fore.CYAN}Press Enter to continue...")
    
    def display_welcome_screen(self):
        """Display a welcome screen with pool theme."""
        self.clear_screen()
        
        # ASCII art for pool
        pool_art = f"""
{Fore.GREEN}  _____       _        _   _               _____           _ 
{Fore.GREEN} |  __ \     | |      | | (_)             |  __ \         | |
{Fore.GREEN} | |__) |___ | |_ __ _| |_ _  ___  _ __   | |__) |__   ___| |
{Fore.GREEN} |  _  // _ \| __/ _` | __| |/ _ \| '_ \  |  ___/ _ \ / _ \ |
{Fore.GREEN} | | \ \ (_) | || (_| | |_| | (_) | | | | | |  | (_) |  __/ |
{Fore.GREEN} |_|  \_\___/ \__\__,_|\__|_|\___/|_| |_| |_|   \___/ \___|_|
        """
        
        print(pool_art)
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'Welcome to Rotation Pool Game Tracker':^60}")
        print(f"{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.WHITE}{'Track your game, scores, and fouls with this interactive tool':^60}")
        print(f"{Fore.CYAN}{'=' * 60}\n")
        
        input(f"{Fore.YELLOW}Press Enter to start the game...")
        self.clear_screen()
    
    def get_player_names(self):
        """Get player names from the user."""
        self.clear_screen()
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'PLAYER SETUP':^50}")
        print(f"{Fore.CYAN}{'=' * 50}")
        
        while True:
            try:
                num_players = int(input(f"{Fore.WHITE}Enter number of players (2-4): "))
                if 2 <= num_players <= 4:
                    break
                else:
                    print(f"{Fore.RED}Please enter a number between 2 and 4.")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.")
        
        player_names = []
        for i in range(num_players):
            name = input(f"{Fore.WHITE}Enter name for Player {i+1}: ")
            player_names.append(name)
        
        return player_names
    
    def display_rollover_applied(self, player):
        """Display message when rollover is applied."""
        print(f"\n{Fore.YELLOW}Roll-over applied. {player} will shoot again.")
        time.sleep(1.5)
    
    def display_skip_applied(self, player):
        """Display message when skip is applied."""
        print(f"\n{Fore.BLUE}Skipped to next player: {player}")
        time.sleep(1.5)
    
    def display_foul_applied(self, player):
        """Display message when a foul is applied."""
        print(f"\n{Fore.RED}Foul applied to {player}")
        time.sleep(1.5) 