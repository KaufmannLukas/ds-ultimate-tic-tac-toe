<template>
    <div class="nav-menu">
        <router-link to="/" custom v-slot="{ navigate }">
            <button @click="navigate">Game</button>
        </router-link>
        <router-link to="/about" custom v-slot="{ navigate }">
            <button @click="navigate">About</button>
        </router-link>
        <router-link to="/rules" custom v-slot="{ navigate }">
            <button @click="navigate">Rules</button>
        </router-link>
        <router-link to="/settings" custom v-slot="{ navigate }">
            <button @click="navigate">Settings</button>
        </router-link>
    </div>
</template>

<style>
.nav-menu {
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: 10.5em;
    margin-bottom: 5em;
}
.nav-menu button {
    margin: 0.5em;
    padding: 0.5em;
    border-radius: 0.5em;
    border: 0.1em solid black;
    background-color: white;
    font-size: 1.5em;
    font-weight: bold;
    cursor: pointer;
}

.nav-menu button:hover {
    background-color: #e6e6e6;
}
</style>