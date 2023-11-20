import './assets/main.css'
import '@shoelace-style/shoelace/dist/themes/light.css';

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { setBasePath } from '@shoelace-style/shoelace/dist/utilities/base-path';
setBasePath('https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.10.0/cdn/');
import Toast from "vue-toastification";
import "vue-toastification/dist/index.css";
import { LoadingPlugin } from 'vue-loading-overlay';
import 'vue-loading-overlay/dist/css/index.css';

import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(Toast, {
    transition: "Vue-Toastification__bounce",
    maxToasts: 20,
    newestOnTop: true
});
app.use(LoadingPlugin);

app.config.ignoredElements = [/^sl-/] // ShoeLace Components

app.mount('#app')
