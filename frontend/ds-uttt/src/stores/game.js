import { defineStore, acceptHMRUpdate } from 'pinia'

const defaultState = () => ({
    "games": {
        "game_0": {
            "won_by": "white",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_1": {
            "won_by": "black",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_2": {
            "won_by": "white",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_3": {
            "won_by": "black",
            "next_move": true,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_4": {
            "won_by": "white",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_5": {
            "won_by": "black",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_6": {
            "won_by": "black",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": false,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_7": {
            "won_by": "white",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        },
        "game_8": {
            "won_by": "black",
            "next_move": false,
            "fields": {
                "field_0": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_1": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_2": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_3": {
                    "white": false,
                    "black": true,
                    "last_move": true,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_4": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_5": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_6": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_7": {
                    "white": false,
                    "black": true,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                },
                "field_8": {
                    "white": true,
                    "black": false,
                    "last_move": false,
                    "blocked_field": true,
                    "valid_move": false
                }
            }
        }
    }
})


export const useGameStore = defineStore('gameStore', {
  state: () => ({
    ...defaultState(),
  }),
  actions: { },
  getters: {
  },
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useGameStore, import.meta.hot))
}