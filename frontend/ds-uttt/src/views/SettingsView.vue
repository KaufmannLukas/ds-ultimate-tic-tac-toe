<template>
  <h1>Settings</h1>
  
  <div class="settings">
  <sl-checkbox 
  :checked="gameStore.settings.show_valid_moves" 
    @click="gameStore.settings.show_valid_moves = $event.target.checked">
    Show valid moves
  </sl-checkbox>

  <sl-checkbox 
  :checked="gameStore.settings.show_local_wins" 
    @click="gameStore.settings.show_local_wins = $event.target.checked">
    Show local wins
  </sl-checkbox>

  <sl-checkbox 
  :checked="gameStore.settings.show_last_move" 
    @click="gameStore.settings.show_last_move = $event.target.checked">
    Show last move
  </sl-checkbox>

  <sl-checkbox 
  :checked="gameStore.settings.show_valid_areas" 
    @click="gameStore.settings.show_valid_areas = $event.target.checked">
    Show valid move area
  </sl-checkbox>
</div>

  <sl-button class="fancy" @click="startGame">
      {{ gameStore.currentGameId ? 'Reset & Start new game' : 'Start new game' }}
  </sl-button>
</template>

<script setup>
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/checkbox/checkbox.js';
import { useRouter } from 'vue-router';
import { useGameStore } from '@/stores/game';

const router = useRouter(); // Initialize useRouter here
const gameStore = useGameStore();

function startGame() {
  console.log('start game');
  gameStore.newGame();
  router.push({ name: 'Game' }); // Use router directly
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

  .settings {
    display: flex;
    flex-direction: column;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
  }

</style>