import axios from 'axios';
import { defineStore, acceptHMRUpdate } from 'pinia';
import {useLoading} from 'vue-loading-overlay'

const BASE_URL = 'http://127.0.0.1:5000';
const NEW_GAME_ENDPOINT = `${BASE_URL}/new_game`;
const GET_GAME_STATE_ENDPOINT = `${BASE_URL}/get_game_state`;
const PLAY_ENDPOINT = `${BASE_URL}/play`;

const apiService = {
    getNewGame: () => axios.get(NEW_GAME_ENDPOINT),
    getGameState: (gameId) => axios.get(`${GET_GAME_STATE_ENDPOINT}?id=${gameId}`),
    postPlay: (data) => axios.post(PLAY_ENDPOINT, data),
};

export const useGameStore = defineStore('gameStore', {
    state: () => ({
        gameState: {},
        baseURL: BASE_URL,
        currentGameId: null,
        playerColor: 'white (X)',
        isLoading: false,
        settings: { 
            show_valid_moves: true,
            show_local_wins: true,
            show_last_move: true,
            show_valid_areas: true,
        },
    }),
    actions: {
        async newGame() {
            const overlay = this._loadingOverlay().show();
            try {
                const response = await apiService.getNewGame();
                this.currentGameId = response.data.game_id;
                this.gameState = response.data.game_state;
                overlay.hide();
            } catch (error) {
                console.error('Error fetching new game:', error);
                overlay.hide();
            }
        },
        async updateGameState() {
            const overlay = this._loadingOverlay().show();
            try {
                let response;
                do {
                    await new Promise(r => setTimeout(r, 1000));
                    response = await apiService.getGameState(this.currentGameId);
                    this.gameState = response.data.game_state;
                } while (response && response.data.agent_is_busy === true && response.data.global_win === "None");
                overlay.hide();
            } catch (error) {
                console.error('Error updating game state:', error);
                overlay.hide();
            }
        },
        async makeMove(gameIndex, fieldIndex) {
            const overlay = this._loadingOverlay().show();
            const data = {
                game_id: this.currentGameId,
                game_idx: gameIndex,
                field_idx: fieldIndex,
            };
            try {
                const response = await apiService.postPlay(data);
                this.gameState = response.data.game_state;
                this.updateGameState();
                overlay.hide();
            } catch (error) {
                console.error('Error making a move:', error);
                overlay.hide();
            }
        },
        _loadingOverlay() {
            const $loading = useLoading({
                color: '#000000',
                opacity: 0.3,
                loader: 'bars'
            });
            return $loading;
        }
    },
    getters: {
        // Add any getters if necessary
    },
});

if (import.meta.hot) {
    import.meta.hot.accept(acceptHMRUpdate(useGameStore, import.meta.hot));
}
