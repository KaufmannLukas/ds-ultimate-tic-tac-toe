<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <header>x
    <router-link to="/">Game</router-link>
    <router-link to="/settings">Settings</router-link>
  </header>

  <RouterView />
</template>


