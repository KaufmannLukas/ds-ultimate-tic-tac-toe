<template>
    <h1>Settings</h1>
    <p>Hier kommen die Game Einstellungen hin </p>
    <sl-button class="fancy" @click="startGame()">Start Game</sl-button>
</template>


<script setup>
import '@shoelace-style/shoelace/dist/components/button/button.js'
import { useRouter } from 'vue-router'
import { useGameStore } from '@/stores/game';

const gameStore = useGameStore();

function startGame() {
    console.log('start game');
    gameStore.newGame();
    const router = useRouter();
    router.push({ name: 'GameView' });
}

</script>

<style>
sl-button.fancy::part(base) {
    /* Set design tokens for height and border width */
    --sl-input-height-medium: 48px;
    --sl-input-border-width: 4px;
    margin: 5px;
    border-radius: 0;
    font-size: 1.125rem;
    box-shadow: 0 2px 10px #0002;
    transition: var(--sl-transition-medium) transform ease, var(--sl-transition-medium) border ease;
  }

  sl-button.fancy::part(base):hover {
    transform: scale(1.05) rotate(-1deg);
  }

  sl-button.fancy::part(base):active {
    color: black;
    transform: scale(1.05) rotate(-1deg) translateY(2px);
  }

  sl-button.fancy::part(base):focus-visible {
    outline: dashed 2px deeppink;
    outline-offset: 4px;
  }

</style>