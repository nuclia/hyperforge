import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import WorkflowEditorPage from './pages/WorkflowEditorPage.vue'
import ChatPage from './pages/ChatPage.vue'
import DriversPage from './pages/DriversPage.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', component: WorkflowEditorPage },
    { path: '/chat', component: ChatPage },
    { path: '/sources', component: DriversPage },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
