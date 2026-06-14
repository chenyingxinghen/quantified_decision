import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import AutoImport from 'unplugin-auto-import/vite'
import compression from 'vite-plugin-compression'

export default defineConfig({
    base: '/quant/',
    plugins: [
        vue(),
        AutoImport({
            imports: ['vue', 'vue-router'],
            resolvers: [ElementPlusResolver()],
        }),
        Components({
            resolvers: [ElementPlusResolver({ importStyle: 'css' })],
        }),
        compression({ algorithm: 'gzip', ext: '.gz', threshold: 10240 }),
        compression({ algorithm: 'brotliCompress', ext: '.br', threshold: 10240 }),
    ],
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
                    'vue-core': ['vue', 'vue-router'],
                },
            },
        },
        chunkSizeWarningLimit: 600,
    },
})
