<script setup>
import { ref } from 'vue';
import subGame from '@/components/subGame.vue';


// options:
// 9 games: local win (X/O), valid moves / where to play next (one or more games possible)
// 81 fields: last move (show or not?), black or white stone.

const currentCell = ref(null);
const currentGame = ref(null);


function subHover(game_id, cell_id) {
    currentGame.value = game_id + 1;
    currentCell.value = cell_id + 1;
}

const gameState = ref({
    game1: {},
    game2: {highlight: true},
    game3: {},
    game4: {},
    game5: {},
    game6: {},
    game7: {highlight: true},
    game8: {},
    game9: {}
})


</script>
ss
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
</style>

<template>
    <main>
        <div class="display">Game: {{ currentGame }}, Cell: {{ currentCell }}</div>
        <div class="game_wrapper">
            <div class="grid" id="main_game">
                <subGame v-for="_, index in 9" 
                :game_id="index" 
                :key="index" 
                :passHover="subHover"
                :gameState="gameState['game' + (index + 1)]" />
            </div>
        </div>
    </main>
</template>
