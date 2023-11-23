<template>
    <div class="cell" :id="game_id" :class="gameClasses">
        <div class="sub-cell" v-for="cell in cells" :key="cell.id" @mouseover="passHover(props.game_id, cell.id)"
            @click="handleCellClick(cell.id)" @mouseenter="setHovered(cell.id, true)"
            @mouseleave="setHovered(cell.id, false)" :class="getCellClass(cell.id)">
            <img v-if="isStonePresent(cell.id, 'black')" src="@/assets/icon_black_stone.png" alt="black stone" />
            <img v-if="isStonePresent(cell.id, 'white')" src="@/assets/icon_white_stone.png" alt="white stone" />
            <img v-if="isHovered[cell.id] && isLegalMove(cell.id) && settings.show_valid_moves" src="@/assets/icon_white_stone.png"
                alt="white stone half opacity" style="opacity: 0.5;" />
        </div>
    </div>
</template>

<script setup>
import { ref, computed, reactive  } from 'vue';
import { useGameStore } from '@/stores/game';
import { useToast } from "vue-toastification";
const toast = useToast();

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

const cells = ref(Array.from({ length: 9 }, (_, i) => ({ id: i })));
const isHovered = reactive({});
const gameStore = useGameStore();

const gameClasses = computed(() => {
    const game = gameStore.gameState.games[`game_${props.game_id}`];
    const isAllValid = gameStore.gameState.games.game_0.next_move && gameStore.gameState.games.game_1.next_move && gameStore.gameState.games.game_2.next_move && gameStore.gameState.games.game_3.next_move && gameStore.gameState.games.game_4.next_move && gameStore.gameState.games.game_5.next_move && gameStore.gameState.games.game_6.next_move && gameStore.gameState.games.game_7.next_move && gameStore.gameState.games.game_8.next_move;
    console.log('isAllValid', isAllValid);
    return {
        'highlight-game': game?.next_move && settings.value.show_valid_areas && !isAllValid,
        'won_by_black': game?.won_by === 'black' && settings.value.show_local_wins,
        'won_by_white': game?.won_by === 'white' && settings.value.show_local_wins,
        'won_by_draw': game?.won_by === 'draw' && settings.value.show_local_wins
    };
});

const settings = computed(() => {
    return gameStore.settings;
});

function handleCellClick(cell_id) {
    const game = gameStore.gameState.games[`game_${props.game_id}`];
    if (game?.fields[`field_${cell_id}`]?.valid_move) {
        makeMove(cell_id);
    } else {
        showToast(`Illegal move! You cannot make a move in game ${props.game_id + 1}, cell ${cell_id + 1} now!`, 'info')
    }
}

function setHovered(cellId, value) {
    isHovered[cellId] = value;
}

function makeMove(cell_id) {
    console.log('make move');
    gameStore.makeMove(props.game_id, cell_id);
}

function getCellClass(cell_id) {
    const game = gameStore.gameState.games[`game_${props.game_id}`];
    return {
        'highlight-last-move': game?.fields[`field_${cell_id}`]?.last_move && settings.value.show_last_move,
    };
}

function isStonePresent(cell_id, color) {
    const game = gameStore.gameState.games[`game_${props.game_id}`];
    return game?.fields[`field_${cell_id}`]?.[color];
}

function isLegalMove(cell_id) {
    const game = gameStore.gameState.games[`game_${props.game_id}`];
    return game?.fields[`field_${cell_id}`]?.valid_move;
}

function showToast(msg, type="info") {
    toast[type](msg, {
        position: "top-center",
        timeout: 3000,
        closeOnClick: true,
        pauseOnFocusLoss: true,
        pauseOnHover: true,
        draggable: true,
        draggablePercent: 0.6,
        showCloseButtonOnHover: false,
        hideProgressBar: true,
        closeButton: "button",
        icon: true,
        rtl: false
    });
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