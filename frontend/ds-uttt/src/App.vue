<script setup>
import { RouterLink, RouterView } from 'vue-router'
import '@shoelace-style/shoelace/dist/components/button-group/button-group.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/dropdown/dropdown.js';
import '@shoelace-style/shoelace/dist/components/menu/menu.js';
import '@shoelace-style/shoelace/dist/components/menu-item/menu-item.js';
</script>

<template>
  <header>
    <div class="navigation">
      <sl-button-group label="navigation">
      <router-link to="/" custom v-slot="{ navigate }">
        <sl-button @click="navigate">Game</sl-button>
      </router-link>
    </sl-button-group>
    <div class="game_name">UTTT</div>
    <div class="dropdown-selection-alt">
  <sl-dropdown>
    <sl-button slot="trigger" caret>Menu</sl-button>
    <sl-menu>
      <router-link to="/about" custom v-slot="{ navigate }">
        <sl-menu-item  @click="navigate">About</sl-menu-item>
      </router-link>
      <router-link to="/rules" custom v-slot="{ navigate }">
        <sl-menu-item  @click="navigate">Rules</sl-menu-item>
      </router-link>
      <router-link to="/settings" custom v-slot="{ navigate }">
        <sl-menu-item  @click="navigate">Settings</sl-menu-item>
      </router-link>
      
    </sl-menu>
  </sl-dropdown>
</div>
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

