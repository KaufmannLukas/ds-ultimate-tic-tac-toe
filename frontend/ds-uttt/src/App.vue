<script setup>
import { RouterView } from 'vue-router'
import { computed } from 'vue';
import '@shoelace-style/shoelace/dist/components/button-group/button-group.js';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/dropdown/dropdown.js';
import '@shoelace-style/shoelace/dist/components/menu/menu.js';
import '@shoelace-style/shoelace/dist/components/menu-item/menu-item.js';

import { useRoute, useRouter } from 'vue-router';
const router = useRouter();
const route = useRoute();
const isMenuRoute = computed(() => route.path === '/menu');

function handleMenu() {
  if (isMenuRoute.value) {
    router.go(-1);
  } else {
    router.push('/menu');
  }
}


</script>

<template>
  <div>
    
  </div>
  <header>
    <div class="navigation">
      <div class="game_name">U_T-T-T</div>
      <div class="dropdown-selection-alt">
        <div class="menu_container" @click="handleMenu()" :class="isMenuRoute ? 'change' : ''">
            <div class="bar1"></div>
            <div class="bar2"></div>
            <div class="bar3"></div>
          </div>
      </div>
    </div>

  </header>

  <RouterView />
  <div class="group_name">XOXO&#178;</div>
</template>

<style>
.navigation {
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-width: 914px;
}

.game_name {
  display: absolute;
  bottom: 0;
  font-size: 3em;
  text-align: center;
  margin: 0.1em;
  color: rgb(0, 0, 0);
}

.group_name {
  font-size: 3em;
  text-align: center;
  color: rgb(0, 0, 0);
}

.menu_container {
  display: inline-block;
  cursor: pointer;
}

.bar1,
.bar2,
.bar3 {
  width: 35px;
  height: 5px;
  background-color: #333;
  margin: 6px 0;
  transition: 0.4s;
}

.change .bar1 {
  transform: translate(0, 11px) rotate(-45deg);
}

.change .bar2 {
  opacity: 0;
}

.change .bar3 {
  transform: translate(0, -11px) rotate(45deg);
}
</style>

