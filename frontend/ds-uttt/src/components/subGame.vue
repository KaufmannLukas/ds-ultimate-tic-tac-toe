<template>
    <div class="cell" :id="game_id" :class="{'highlight-game': gameStore.gameState.games['game_' + (props.game_id)]?.next_move, 
        'won_by_black': gameStore.gameState.games['game_' + (props.game_id)]?.won_by == 'black', 
        'won_by_white': gameStore.gameState.games['game_' + (props.game_id)]?.won_by == 'white',
        'won_by_draw': gameStore.gameState.games['game_' + (props.game_id)]?.won_by == 'draw'
        }">
        <!-- 9 Unter-DIVs pro Haupt-DIV -->
        <div class="sub-cell" 
        v-for="cell in cells" 
        :key="cell.id" 
        @mouseover="passHover(props.game_id, cell.id)"
        @click="() => gameStore.gameState.games['game_' + (props.game_id)]?.fields['field_' + cell.id].valid_move ? makeMove(cell.id) : console.log('invalid move')"
        :class="{'highlight-last-move': gameStore.gameState.games['game_' + (props.game_id)]?.fields['field_' + cell.id].last_move}"
        >
            <img v-if="gameStore.gameState.games['game_' + (props.game_id)]?.fields['field_' + cell.id].black" src="@/assets/icon_black_stone.png" alt="black stone" />
            <img v-if="gameStore.gameState.games['game_' + (props.game_id)]?.fields['field_' + cell.id].white" src="@/assets/icon_white_stone.png" alt="white stone" />
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useGameStore } from '@/stores/game';
import { storeToRefs } from 'pinia';

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

onMounted(() =>{

});

function makeMove(cell_id) {
    console.log('make move');
    gameStore.makeMove(props.game_id, cell_id);
}
    
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

.highlight-game {
   /* glow and flash effect */
    border: 4px solid rgb(129, 192, 255);
   /* animation: glow 1s ease-in-out infinite alternate; */
}

.highlight-last-move {
   /* glow and flash effect */
   /*animation: glow 1s ease-in-out infinite alternate; */
    filter: drop-shadow(0 0 1rem rgb(63, 159, 255));
}


.won_by_black {
    /* put an image in the background with 50% opacity */
    background-image: url(@/assets/icon_local_win_O_0.8.png);
    background-repeat: no-repeat;
    background-position: center;
    background-size: 100%;
    opacity: 1;
}

.won_by_white {
    background-image: url(@/assets/icon_local_win_X_0.8.png);
    background-repeat: no-repeat;
    background-position: center;
    background-size: 100%;
    opacity: 1;
}

.won_by_draw {
    background-image: url(@/assets/icon_local_draw_0.8.png);
    background-repeat: no-repeat;
    background-position: center;
    background-size: 100%;
    opacity: 1;
}

@keyframes glow {
    from {
        filter: drop-shadow(0 0 1rem rgb(254, 255, 254));
    }
    to {
        filter: drop-shadow(0 0 1rem rgb(129, 192, 255));
    }
}
</style>