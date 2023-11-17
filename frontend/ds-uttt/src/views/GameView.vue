<script setup>
import { ref } from 'vue';


// options:
// 9 games: local win (X/O), valid moves / where to play next (one or more games possible)
// 81 fields: last move (show or not?), black or white stone.

const currentCell = ref(0);
const currentGame = ref(0);




</script>

<style>

.sub-cell img { /* Figures on the board */
    width: 100%;
    height: 100%;
    object-fit:cover;
    /* center the image */
    display: block;
}

.sub-cell:hover {
    /* make it a little transparent */
    opacity: 0.5;
}


.game_wrapper {
    display: flex;
    justify-content: center;
}

.display {
    display: flex;
    flex-direction: row;
    justify-content: center;
    width: 100%;
}
</style>

<template>
    <main>
        <div class="display">Game: 1, Cell: 5</div>
        <div class="game_wrapper">
            <div class="grid" id="main_game">
                <!-- 9 Haupt-DIVs -->
                <div class="cell" id="game_1">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"><img src="@/assets/icon_white_stone.png"></div>
                    <div class="sub-cell"><img src="@/assets/icon_black_stone.png"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_2">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_3">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_4">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_5">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_6">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_7">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_8">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <div class="cell" id="game_9">
                    <!-- 9 Unter-DIVs pro Haupt-DIV -->
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                    <div class="sub-cell"></div>
                </div>
                <!-- Wiederhole das obige 8 Mal -->
                <!-- ... -->
            </div>
        </div>
    </main>
</template>
