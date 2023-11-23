<template>
  <div class="settings-view">
    <h1>Settings</h1>
    <sl-dialog ref='confirmDialog' label="Warning" class="dialog-deny-close">
      Are you sure you want start a new game? <br/>
      
      <sl-button slot="footer" variant="danger"  @click="startGame">Yes</sl-button>
    </sl-dialog>

    <div class="settings">
      <p>choose player</p>
      <sl-select>
        <sl-option selected value="white"><img style="vertical-align: bottom;" src="@/assets/icon_white_stone.png"
            width="20"> white</sl-option>
        <sl-option value="black"><img style="vertical-align: bottom;" src="@/assets/icon_black_stone.png" width="20">
          black</sl-option>
      </sl-select>
      <div class="switch-container">
        <sl-switch :checked="gameStore.settings.show_valid_moves"
          @click="gameStore.settings.show_valid_moves = $event.target.checked">
          show valid moves
        </sl-switch>

        <sl-switch :checked="gameStore.settings.show_local_wins"
          @click="gameStore.settings.show_local_wins = $event.target.checked">
          show local wins
        </sl-switch>

        <sl-switch :checked="gameStore.settings.show_last_move"
          @click="gameStore.settings.show_last_move = $event.target.checked">
          show last move
        </sl-switch>

        <sl-switch :checked="gameStore.settings.show_valid_areas"
          @click="gameStore.settings.show_valid_areas = $event.target.checked">
          show valid move area
        </sl-switch>
      </div>

    </div>

    <button @click="() => { showDialog() }">
      {{ gameStore.currentGameId != null ? 'Reset game' : 'Start new game' }}
    </button>
  </div>
</template>

<script setup>
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/switch/switch.js';
import '@shoelace-style/shoelace/dist/components/select/select.js';
import '@shoelace-style/shoelace/dist/components/option/option.js';
import '@shoelace-style/shoelace/dist/components/dialog/dialog.js';
import { useRouter } from 'vue-router';
import { useGameStore } from '@/stores/game';
import { ref } from 'vue';

const confirmDialog = ref(null);
const router = useRouter(); // Initialize useRouter here
const gameStore = useGameStore();

function showDialog() {
      if (confirmDialog.value) {
        confirmDialog.value.show();
      }
    }

function startGame() {
  gameStore.newGame();
  router.push({ name: 'Game' }); // Use router directly
}
</script>
<style>
.settings {
  display: flex;
  flex-direction: column;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  font-size: 1.5em;

}

sl-switch {
  margin-top: 0.3em;
  margin-bottom: 0.3em;
  --width: 40px;
  --height: 20px;
  --thumb-size: 18px;
}

sl-switch>div {
  font-size: 2em;
}

.settings-view {

  width: 70%;
  margin-bottom: 5em;
  display: flex;
  flex-direction: column;
  align-items: center;

}

.switch-container {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  font-size: 32px;
}
</style>