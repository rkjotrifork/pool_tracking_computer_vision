document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const scoreboardBody = document.getElementById('scoreboard-body');
    const gameInfoContent = document.getElementById('game-info-content');
    const lastActionDiv = document.getElementById('last-action');
    const currentPlayerSpan = document.getElementById('current-player');
    const lowestBallSpan = document.getElementById('lowest-ball');
    
    // Buttons
    const btnApplyFoul = document.getElementById('btn-apply-foul');
    const btnApplyRollover = document.getElementById('btn-apply-rollover');
    const btnSkipPlayer = document.getElementById('btn-skip-player');
    const btnNewGame = document.getElementById('btn-new-game');
    
    // Initial load
    updateGameState();
    
    // Event listeners
    btnApplyFoul.addEventListener('click', applyFoul);
    btnApplyRollover.addEventListener('click', applyRollover);
    btnSkipPlayer.addEventListener('click', skipPlayer);
    btnNewGame.addEventListener('click', confirmNewGame);
    
    // Functions
    function updateGameState() {
        fetch('/api/game_state')
            .then(response => response.json())
            .then(data => {
                updateScoreboard(data.players_data, data.player_order);
                updateGameInfo(data);
                updateLastAction(data.last_action);
                
                // Disable buttons if game is over
                if (data.game_over) {
                    btnApplyFoul.disabled = true;
                    btnApplyRollover.disabled = true;
                    btnSkipPlayer.disabled = true;
                } else {
                    btnApplyFoul.disabled = false;
                    
                    // Only enable rollover if the previous player committed a foul
                    // Check if the last action contains "foul" or "Foul"
                    const lastActionLower = data.last_action.toLowerCase();
                    btnApplyRollover.disabled = !(lastActionLower.includes('foul') && !lastActionLower.includes('reset'));
                    
                    btnSkipPlayer.disabled = false;
                }
            })
            .catch(error => console.error('Error fetching game state:', error));
    }
    
    function updateScoreboard(playersData, playerOrder) {
        scoreboardBody.innerHTML = '';
        
        playerOrder.forEach(player => {
            const data = playersData[player];
            const row = document.createElement('tr');
            
            if (data.status === 'ACTIVE') {
                row.classList.add('active');
            } else if (data.status === 'NEXT') {
                row.classList.add('next');
            }
            
            // Create player name cell with pocketed balls
            const nameCell = document.createElement('td');
            nameCell.className = 'player-name-cell';
            
            // Create the player name element
            const playerName = document.createElement('span');
            playerName.textContent = player;
            nameCell.appendChild(playerName);
            
            // Add pocketed balls if any
            if (data.pocketed_balls && data.pocketed_balls.length > 0) {
                const pocketedBallsContainer = document.createElement('div');
                pocketedBallsContainer.className = 'pocketed-balls-container';
                
                data.pocketed_balls.forEach(ball => {
                    let ballClass = '';
                    if (ball <= 8) {
                        ballClass = `solid solid-${ball}`;
                    } else {
                        ballClass = `striped striped-${ball}`;
                    }
                    
                    const ballElement = document.createElement('div');
                    ballElement.className = `pool-ball mini-small ${ballClass}`;
                    
                    const numberSpan = document.createElement('span');
                    numberSpan.textContent = ball;
                    ballElement.appendChild(numberSpan);
                    
                    pocketedBallsContainer.appendChild(ballElement);
                });
                
                nameCell.appendChild(pocketedBallsContainer);
            }
            
            // Create score cell with editable functionality
            const scoreCell = document.createElement('td');
            scoreCell.className = 'score-cell';
            scoreCell.innerHTML = `
                <span class="score-display">${data.score}</span>
                <input type="number" class="score-edit" value="${data.score}" style="display: none;">
                <button class="btn-edit-score" title="Edit Score">✏️</button>
            `;
            
            // Create fouls cell with editable functionality
            const foulsCell = document.createElement('td');
            foulsCell.className = 'fouls-cell';
            foulsCell.innerHTML = `
                <span class="fouls-display">${data.fouls}</span>
                <input type="number" class="fouls-edit" value="${data.fouls}" style="display: none;">
                <button class="btn-edit-fouls" title="Edit Fouls">✏️</button>
            `;
            
            row.appendChild(nameCell);
            row.appendChild(scoreCell);
            row.appendChild(foulsCell);
            row.innerHTML += `<td class="status-${data.status.toLowerCase()}">${data.status}</td>`;
            
            scoreboardBody.appendChild(row);
            
            // Add event listeners for score editing
            setupEditableCell(row, '.btn-edit-score', '.score-display', '.score-edit', player, 'score');
            
            // Add event listeners for fouls editing
            setupEditableCell(row, '.btn-edit-fouls', '.fouls-display', '.fouls-edit', player, 'fouls');
        });
    }
    
    // Helper function to set up editable cells
    function setupEditableCell(row, btnSelector, displaySelector, editSelector, player, field) {
        const editBtn = row.querySelector(btnSelector);
        const displayEl = row.querySelector(displaySelector);
        const editEl = row.querySelector(editSelector);
        
        editBtn.addEventListener('click', function() {
            // Toggle between display and edit mode
            if (displayEl.style.display !== 'none') {
                displayEl.style.display = 'none';
                editEl.style.display = 'inline-block';
                editEl.focus();
                editBtn.textContent = '✓';
            } else {
                // Save the new value
                const newValue = parseInt(editEl.value);
                if (!isNaN(newValue)) {
                    if (field === 'score') {
                        updatePlayerScore(player, newValue);
                    } else if (field === 'fouls') {
                        updatePlayerFouls(player, newValue);
                    }
                }
                
                displayEl.style.display = 'inline-block';
                editEl.style.display = 'none';
                editBtn.textContent = '✏️';
            }
        });
        
        // Also save on Enter key
        editEl.addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                const newValue = parseInt(editEl.value);
                if (!isNaN(newValue)) {
                    if (field === 'score') {
                        updatePlayerScore(player, newValue);
                    } else if (field === 'fouls') {
                        updatePlayerFouls(player, newValue);
                    }
                }
                
                displayEl.style.display = 'inline-block';
                editEl.style.display = 'none';
                editBtn.textContent = '✏️';
            }
        });
    }
    
    function updatePlayerScore(player, newScore) {
        fetch('/api/update_score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                player: player,
                score: newScore 
            }),
        })
        .then(response => response.json())
        .then(data => {
            updateGameState();
        })
        .catch(error => console.error('Error updating score:', error));
    }
    
    function updatePlayerFouls(player, newFouls) {
        fetch('/api/update_fouls', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                player: player,
                fouls: newFouls 
            }),
        })
        .then(response => response.json())
        .then(data => {
            updateGameState();
        })
        .catch(error => console.error('Error updating fouls:', error));
    }
    
    function updateGameInfo(data) {
        let html = '';
        
        if (data.game_over) {
            html += `<div class="game-over">
                <h3>Game Over!</h3>
                <p class="winner">Winner: ${findWinner(data.players_data)}</p>
            </div>`;
        } else {
            html += `<div class="game-info-item">
                <span class="game-info-label">Current Player:</span> 
                <span class="status-active">${data.current_player}</span>
            </div>`;
            
            // Display lowest ball as an icon
            const lowestBall = data.lowest_ball;
            let lowestBallClass = '';
            if (lowestBall <= 8) {
                lowestBallClass = `solid solid-${lowestBall}`;
            } else {
                lowestBallClass = `striped striped-${lowestBall}`;
            }
            
            html += `<div class="game-info-item">
                <span class="game-info-label">Lowest Ball:</span> 
                <div class="pool-ball mini ${lowestBallClass}"><span>${lowestBall}</span></div>
            </div>`;
            
            // Display balls on table as interactive icons
            html += `<div class="game-info-item">
                <span class="game-info-label">Balls on Table:</span> 
                <div class="balls-on-table-container" id="interactive-balls">`;
            
            data.balls_on_table.forEach(ball => {
                let ballClass = '';
                if (ball <= 8) {
                    ballClass = `solid solid-${ball}`;
                } else {
                    ballClass = `striped striped-${ball}`;
                }
                
                // Check if this ball is in the selectedBalls array
                const isSelected = selectedBalls.includes(ball);
                const selectedClass = isSelected ? 'selected' : '';
                
                html += `<div class="pool-ball mini interactive ${ballClass} ${selectedClass}" data-ball="${ball}"><span>${ball}</span></div>`;
            });
            
            html += `</div></div>`;
            
            // Add a process shot button
            html += `<div class="game-info-item">
                <div class="game-controls-container">
                    <button id="btnProcessShot" class="btn info">Process Shot</button>
                </div>
            </div>`;
        }
        
        gameInfoContent.innerHTML = html;
        
        // Add event listeners to the interactive balls
        if (!data.game_over) {
            const interactiveBalls = document.querySelectorAll('#interactive-balls .pool-ball.interactive');
            interactiveBalls.forEach(ball => {
                ball.addEventListener('click', toggleBallSelectionDirect);
            });
            
            // Add event listener to the process shot button
            document.getElementById('btnProcessShot').addEventListener('click', processDirectShot);
        }
    }
    
    function updateLastAction(lastAction) {
        if (lastAction) {
            // Determine the color based on the action type
            const lastActionLower = lastAction.toLowerCase();
            
            if (lastActionLower.includes('foul')) {
                // Red for fouls
                lastActionDiv.className = 'alert alert-danger';
            } else if (lastActionLower.includes('pocketed') || lastActionLower.includes('legal shot')) {
                // Green for successful shots
                lastActionDiv.className = 'alert alert-success';
            } else if (lastActionLower.includes('roll-over')) {
                // Orange for rollovers
                lastActionDiv.className = 'alert alert-warning';
            } else if (lastActionLower.includes('next player') || lastActionLower.includes('skipped')) {
                // Blue for turn changes
                lastActionDiv.className = 'alert alert-info';
            } else {
                // Default style
                lastActionDiv.className = 'alert';
            }
            
            // Replace ball numbers with ball icons
            lastActionDiv.innerHTML = replaceBallNumbersWithIcons(lastAction);

            // Make sure we don't replace text in buttons with the no-replace class
            document.querySelectorAll('.no-replace').forEach(el => {
                el.innerHTML = el.innerHTML.replace(/<div class="pool-ball.*?<\/div>/g, 
                    match => el.textContent);
            });
            
            lastActionDiv.style.display = 'block';
        } else {
            lastActionDiv.style.display = 'none';
        }
    }
    
    // Function to replace ball numbers with ball icons
    function replaceBallNumbersWithIcons(text) {
        // Only replace numbers that are surrounded by spaces, commas, brackets, or at the beginning/end of text
        const ballNumberRegex = /(^|\s|,|\[|\()([1-9]|1[0-5])(\s|,|\]|\)|$)/g;
        
        return text.replace(ballNumberRegex, (match, before, number, after) => {
            const ballNumber = parseInt(number);
            let ballClass = '';
            
            if (ballNumber <= 8) {
                ballClass = `solid solid-${ballNumber}`;
            } else {
                ballClass = `striped striped-${ballNumber}`;
            }
            
            return before + `<div class="pool-ball mini ${ballClass}"><span>${ballNumber}</span></div>` + after;
        });
    }
    
    function findWinner(playersData) {
        let winner = '';
        let highestScore = -1;
        
        for (const [player, data] of Object.entries(playersData)) {
            if (data.score > highestScore) {
                highestScore = data.score;
                winner = `${player} with ${data.score} points`;
            }
        }
        
        return winner;
    }
    
    function applyFoul() {
        fetch('/api/apply_foul', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            updateGameState();
        })
        .catch(error => console.error('Error applying foul:', error));
    }
    
    function applyRollover() {
        fetch('/api/apply_rollover', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            updateGameState();
        })
        .catch(error => console.error('Error applying rollover:', error));
    }
    
    function skipPlayer() {
        fetch('/api/skip_player', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            updateGameState();
        })
        .catch(error => console.error('Error skipping player:', error));
    }
    
    function confirmNewGame() {
        if (confirm('Are you sure you want to start a new game? Current game progress will be lost.')) {
            fetch('/api/new_game', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                window.location.href = '/setup';
            })
            .catch(error => console.error('Error starting new game:', error));
        }
    }
    
    // Auto-refresh game state every 5 seconds
    setInterval(updateGameState, 5000);
    
    // Global variable to track selected balls
    let selectedBalls = [];

    function toggleBallSelectionDirect(event) {
        const ballElement = event.currentTarget;
        const ballNumber = parseInt(ballElement.dataset.ball);
        
        // Toggle selection
        if (ballElement.classList.contains('selected')) {
            // Remove from selection
            ballElement.classList.remove('selected');
            selectedBalls = selectedBalls.filter(ball => ball !== ballNumber);
        } else {
            // Add to selection
            ballElement.classList.add('selected');
            selectedBalls.push(ballNumber);
        }
    }

    function processDirectShot() {
        if (selectedBalls.length === 0) {
            alert('Please select at least one ball that was pocketed.');
            return;
        }
        
        fetch('/api/process_shot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pocketed_balls: selectedBalls }),
        })
        .then(response => response.json())
        .then(data => {
            // Clear selection
            selectedBalls = [];
            updateGameState();
        })
        .catch(error => console.error('Error processing shot:', error));
    }
}); 