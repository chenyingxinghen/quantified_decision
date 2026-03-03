import { createRouter, createWebHistory } from 'vue-router'

const routes = [
    {
        path: '/',
        name: 'Dashboard',
        component: () => import('../views/DashboardView.vue'),
        meta: { title: '仪表盘', icon: 'Odometer' },
    },
    {
        path: '/selector',
        name: 'StockSelector',
        component: () => import('../views/StockSelectorView.vue'),
        meta: { title: '选股与信号', icon: 'Search' },
    },
    {
        path: '/paper-trading',
        name: 'PaperTrading',
        component: () => import('../views/PaperTradingView.vue'),
        meta: { title: '实盘验证', icon: 'Briefcase' },
    },

    {
        path: '/analysis',
        name: 'Analysis',
        component: () => import('../views/AnalysisView.vue'),
        meta: { title: '技术分析', icon: 'TrendCharts' },
    },
    {
        path: '/config',
        name: 'Config',
        component: () => import('../views/ConfigView.vue'),
        meta: { title: '配置中心', icon: 'Setting' },
    },
]

const router = createRouter({
    history: createWebHistory('/quant/'), // <--- Added '/quant/'
    routes,
})

export default router
