import './assets/main.css'
import '@shoelace-style/shoelace/dist/themes/light.css';

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { setBasePath } from '@shoelace-style/shoelace/dist/utilities/base-path'; 
setBasePath('https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.10.0/cdn/');

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.config.ignoredElements = [/^sl-/] // ShoeLace Components

app.mount('#app')
