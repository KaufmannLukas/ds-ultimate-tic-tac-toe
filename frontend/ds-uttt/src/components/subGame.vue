<template>
    <div class="cell" :id="game_id" :class="{'highlight': currentGame?.next_move}">
        <!-- 9 Unter-DIVs pro Haupt-DIV -->
        <div class="sub-cell" 
        v-for="cell in cells" 
        :key="cell.id" 
        @mouseover="passHover(props.game_id, cell.id)"
        @click="() => currentGame?.fields['field_' + cell.id].validMove ? console.log('valid move') : console.log('invalid move')"
        :class="{'highlight': currentGame?.fields['field_' + cell.id].last_move}"
        >
            <img v-if="currentGame?.fields['field_' + cell.id].black" src="@/assets/icon_black_stone.png" alt="black stone" />
            <img v-if="currentGame?.fields['field_' + cell.id].white" src="@/assets/icon_white_stone.png" alt="white stone" />
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useGameStore } from '@/stores/game';

const props = defineProps({
    game_id: {
        type: Number,
        required: true
    },
    passHover: {
        type: Function,
        required: true
    },
});

const cells = ref([
    {id: 0},
    {id: 1},
    {id: 2},
    {id: 3},
    {id: 4},
    {id: 5},
    {id: 6},
    {id: 7},
    {id: 8}]);

const gameStore = useGameStore();
const currentGame = ref(null);

onMounted(() =>{
    currentGame.value = gameStore.games['game_' + (props.game_id)];
    console.log(currentGame.value);
});

    
function getStone(type) {
    if (type == 'black') {
        return 'icon_black_stone.png';
    } else if (type == 'white') {
        return 'icon_white_stone.png';
    } else {
        return '';
    }
}

</script>


<style>
.sub-cell img {
    /* Figures on the board */
    width: 100%;
    height: 100%;
    object-fit: cover;
    /* center the image */
    display: block;
}

.sub-cell:hover {
    /* make it a little transparent */
    opacity: 0.5;
}

.game-highlight {
    border: 2px solid rgb(11, 231, 11);
}

.highlight {
   /* glow and flash effect */
    animation: glow 1s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        filter: drop-shadow(0 0 1rem rgb(9, 100, 9));
    }
    to {
        filter: drop-shadow(0 0 1rem rgb(11, 231, 11));
    }
}
</style>