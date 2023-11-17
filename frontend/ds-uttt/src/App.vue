<script setup>
import { RouterLink, RouterView } from 'vue-router'
import '@shoelace-style/shoelace/dist/components/button-group/button-group.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
</script>

<template>
  <header>
    <div class="navigation">
      <sl-button-group label="navigation">
      <router-link to="/" custom v-slot="{ navigate }">
        <sl-button @click="navigate">Game</sl-button>
      </router-link>
      <router-link to="/settings" custom v-slot="{ navigate }">
        <sl-button @click="navigate">Settings</sl-button>
      </router-link>
    </sl-button-group>
    <div class="game_name">UTTT</div>
    </div>
    
  </header>

  <RouterView />
</template>

<style>
.navigation {
  display: flex;
  justify-content:center;
  align-items: center;
}

.game_name {
    padding-left: 0.5em;
    font-size: 4em;
    text-align: center;
    margin: 0.1em;
    color: rgb(144, 86, 15);
    transform: rotate(-5deg);
}


</style>

