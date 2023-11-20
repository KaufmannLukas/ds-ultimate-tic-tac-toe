<script setup>
import '@shoelace-style/shoelace/dist/components/alert/alert.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/icon/icon.js';
import { ref } from 'vue';
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

</script>


<template>
    <main>
        <div class="display">Game: {{ currentGame }} | Cell: {{ currentCell }}</div>
        <div class="alert-display">
            <sl-alert variant="primary" duration="3000" closable>
                <sl-icon slot="icon" name="info-circle"></sl-icon>
                {{  }}
            </sl-alert>
        </div>
        <div class="game_wrapper">
            <div class="grid" id="main_game" v-if="gameStore.currentGameId != null">
                <subGame v-for="_, index in 9" :game_id="index" :key="index" :passHover="subHover"/>
            </div>
            <div v-else>
                <sl-alert variant="primary" open>
                    <sl-icon slot="icon" name="info-circle"></sl-icon>
                    <p>Start a game via "Settings"</p>
                </sl-alert>

            </div>
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
</style>