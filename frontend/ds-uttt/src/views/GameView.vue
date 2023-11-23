<script setup>
import '@shoelace-style/shoelace/dist/components/alert/alert.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/icon/icon.js';
import { computed, ref } from 'vue';
import subGame from '@/components/subGame.vue';
import { useGameStore } from '@/stores/game';

const gameStore = useGameStore();

// options:
// 9 games: local win (X/O), valid moves / where to play next (one or more games possible)
// 81 fields: last move (show or not?), black or white stone.

const currentCell = ref("?");
const currentGame = ref("?");


function subHover(game_id, cell_id) {
    currentGame.value = game_id + 1;
    currentCell.value = cell_id + 1;
}

function startGame() {
    console.log('start game');
    gameStore.newGame();
}

const gameInfo = computed(() => {
    if (gameStore.currentGameId == null) {
        return "";
    }

    const gameState = gameStore.gameState;
    
    if (gameState.global_win) {
        if (gameState.global_win === 'black') {
            return "black (O) won!";
        } else if (gameState.global_win === 'white') {
            return "white (X) won!";
        } else if (gameState.global_win === 'draw') {
            return "draw!";
        }
    }

    if (gameState.current_player) {
        if (gameState.current_player === 'white (X)') {
            return "white (X) to move";
        } else if (gameState.current_player === 'black (O)') {
            return "black (O) to move";
        }
    }

    return "unknown state";
});


const gameClasses = computed(() => {
    return {
        'highlight-whole-game': gameStore.gameState.games.game_0.next_move && gameStore.gameState.games.game_1.next_move && gameStore.gameState.games.game_2.next_move && gameStore.gameState.games.game_3.next_move && gameStore.gameState.games.game_4.next_move && gameStore.gameState.games.game_5.next_move && gameStore.gameState.games.game_6.next_move && gameStore.gameState.games.game_7.next_move && gameStore.gameState.games.game_8.next_move,
    }
});

</script>


<template>
    <main>
        <div class="info-display">
            {{ gameInfo }}
        </div>
        <div v-if="false" class="display">Game: {{ currentGame }} | Cell: {{ currentCell }}</div>
        <div class="game_wrapper">
            <div class="grid" id="main_game" v-if="gameStore.currentGameId != null" :class="gameClasses">
                <subGame v-for="_, index in 9" :game_id="index" :key="index" :passHover="subHover" />
            </div>
        </div>
        <sl-button class="fancy" @click="startGame" v-if="gameStore.currentGameId == null">
            {{ gameStore.currentGameId ? 'Reset & Start new game' : 'Start new game' }}
        </sl-button>
        <div class="stats-display" v-if="gameStore.currentGameId != null">
            <div>white: {{ gameStore.gameState.local_wins_white }}</div>
            <div>black: {{ gameStore.gameState.local_wins_black }}</div>
        </div>
    </main>
</template>

<style>
.game_wrapper {
    display: flex;
    justify-content: center;
}


.display {
    display: flex;
    flex-direction: row;
    justify-content: center;
    width: 100%;
    font-size: large;
}

.info-display {
    display: flex;
    flex-direction: row;
    justify-content: center;
    width: 100%;
    font-size: 2.7em;
    margin-bottom: 0.3em;
}

.stats-display {
    margin-top: 0.3em;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    width: 100%;
    font-size: 2.7em;

}
</style>