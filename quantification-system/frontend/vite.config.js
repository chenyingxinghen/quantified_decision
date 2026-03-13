import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
    base: '/quant/',
    plugins: [vue()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    server: {
        port: 5173,
        proxy: {
            '/quant/api': {
                target: 'http://localhost:8083',
                changeOrigin: true,
            },
        },
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: {
                    'echarts': ['echarts'],
                    'element-plus': ['element-plus'],
                    'vue-core': ['vue', 'vue-router']
                }
            }
        },
        chunkSizeWarningLimit: 1500
    }
})


