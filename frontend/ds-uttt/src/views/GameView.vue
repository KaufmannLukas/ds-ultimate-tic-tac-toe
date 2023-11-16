<script setup>

// options:
// 9 games: local win (X/O), valid moves / where to play next (one or more games possible)
// 81 fields: last move (show or not?), black or white stone.





</script>

<template>
  <main>
  <h1>UTTT</h1>
  <div class="game_wrapper">
  <div class="grid" id="main_game">
        <!-- 9 Haupt-DIVs -->
        <div class="cell" id="game_1" >
            <!-- 9 Unter-DIVs pro Haupt-DIV -->
            <div @click="makeMove(1,1)" class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
            <div class="sub-cell"></div>
        </div>
        <div class="cell" id="game_2" >
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
        <div class="cell" id="game_3" style="background: red;">
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
        <div class="cell" id="game_4" >
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
