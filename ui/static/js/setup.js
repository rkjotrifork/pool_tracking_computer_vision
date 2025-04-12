document.addEventListener('DOMContentLoaded', function() {
    const addPlayerBtn = document.getElementById('add-player');
    const playerInputs = document.getElementById('player-inputs');
    let playerCount = 2;
    
    addPlayerBtn.addEventListener('click', function() {
        playerCount++;
        
        const newPlayerGroup = document.createElement('div');
        newPlayerGroup.className = 'form-group';
        
        newPlayerGroup.innerHTML = `
            <label for="player${playerCount}">Player ${playerCount}:</label>
            <input type="text" id="player${playerCount}" name="player${playerCount}">
            <button type="button" class="btn-remove-player danger">×</button>
        `;
        
        playerInputs.appendChild(newPlayerGroup);
        
        // Add event listener to the remove button
        const removeBtn = newPlayerGroup.querySelector('.btn-remove-player');
        removeBtn.addEventListener('click', function() {
            playerInputs.removeChild(newPlayerGroup);
        });
    });
}); 