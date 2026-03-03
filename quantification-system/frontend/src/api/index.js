import axios from 'axios'
import { ElMessage } from 'element-plus'

const api = axios.create({
    baseURL: '/quant/api',
    timeout: 120000,
})

// ── 防止重复提交与注入 Token ────────────────────────────────────────
const pendingRequests = new Map()

api.interceptors.request.use(config => {
    // 注入 Token
    const token = localStorage.getItem('quant_user_token')
    if (token) {
        config.headers['Token'] = token
    }

    // 防止重复点击 (只针对非 GET 请求)
    if (['post', 'put', 'delete'].includes(config.method.toLowerCase())) {
        const key = `${config.method}:${config.url}:${JSON.stringify(config.data || {})}`
        const now = Date.now()

        if (pendingRequests.has(key) && (now - pendingRequests.get(key) < 1500)) {
            ElMessage.warning('操作过于频繁，请稍后再试')
            return Promise.reject(new Error('操作过于频繁，请稍后再试'))
        }
        pendingRequests.set(key, now)
    }
    return config
}, error => Promise.reject(error))

api.interceptors.response.use(
    response => response,
    error => {
        if (error.response && error.response.status === 429) {
            ElMessage.error(error.response.data.detail || '请求过于频繁，请稍后再试')
        }
        if (error.response && error.response.status === 401 && error.config.url.includes('/login')) {
            // Let login logic handle this
        } else if (error.response && error.response.status === 401) {
            ElMessage.error('登录失效或无权限，请重新登录')
            localStorage.removeItem('quant_user_token')
            localStorage.removeItem('quant_username')
            // window.location.reload()
        }
        return Promise.reject(error)
    }
)

// ── 本地缓存实现 ─────────────────────────────────────
const CACHE_PREFIX = 'quant_cache_'
const DEFAULT_TTL = 5 * 60 * 1000 // 5 minutes

const getCache = (key) => {
    const item = localStorage.getItem(CACHE_PREFIX + key)
    if (!item) return null
    try {
        const { value, expiry } = JSON.parse(item)
        if (Date.now() > expiry) {
            localStorage.removeItem(CACHE_PREFIX + key)
            return null
        }
        return value
    } catch (e) { return null }
}

const setCache = (key, value, ttl = DEFAULT_TTL) => {
    const item = {
        value,
        expiry: Date.now() + ttl
    }
    localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(item))
}

// 缓存包装器
const cachedGet = async (url, config = {}, ttl = DEFAULT_TTL) => {
    const cacheKey = url + JSON.stringify(config.params || {})
    const cachedData = getCache(cacheKey)
    if (cachedData) return { data: cachedData, fromCache: true }

    const response = await api.get(url, config)
    setCache(cacheKey, response.data, ttl)
    return response
}

// ── 选股与信号 ──────────────────────────────────────
export const stockSelector = {
    getLatest: () => api.get('/stock-selector/latest'),
    getModels: () => api.get('/stock-selector/models'),
    runSelection: (params) => api.post('/stock-selector/run', params),
    getSelectionStatus: () => api.get('/stock-selector/run-status'),
    getFactors: (code, days = 250) => cachedGet(`/stock-selector/factors/${code}`, { params: { days } }),
    getSignals: (code, days = 100) => cachedGet(`/stock-selector/signals/${code}`, { params: { days } }),
}

// ── 实盘验证 (支持游客本地存储) ────────────────────────────────
export const paperTrading = {
    getPositions: async (status = 'active') => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            let filtered = positions
            if (status === 'active') filtered = positions.filter(p => p.status === 'active')
            else if (status === 'closed') filtered = positions.filter(p => p.status === 'closed')

            // 为活跃持仓注入最新价格 (尝试从后端获取，如果失败则保持原样)
            if (status !== 'closed') {
                for (let p of filtered) {
                    try {
                        // 借用分析接口获取价格
                        const res = await api.get(`/paper-trading/check-exit/${p.code}`, {
                            params: { buy_price: p.buy_price || 0, buy_date: p.buy_date }
                        })
                        p.latest_price = res.data.current_price
                        if (p.buy_price) {
                            p.unrealized_pct = parseFloat(((p.latest_price - p.buy_price) / p.buy_price * 100).toFixed(2))
                        }
                    } catch (e) { }
                }
            }
            return { data: { positions: filtered.sort((a, b) => b.id - a.id) } }
        }
        return api.get('/paper-trading/positions', { params: { status } })
    },
    buy: async (data) => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            const newPos = {
                ...data,
                id: Date.now(),
                status: 'active',
                created_at: new Date().toLocaleString()
            }
            positions.push(newPos)
            localStorage.setItem('guest_positions', JSON.stringify(positions))
            return { data: { id: newPos.id, message: '已保存至本地 (游客模式)' } }
        }
        return api.post('/paper-trading/buy', data)
    },
    sell: async (data) => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            const idx = positions.findIndex(p => p.id === data.position_id)
            if (idx !== -1) {
                const p = positions[idx]
                p.status = 'closed'
                p.sell_date = data.sell_date
                p.sell_price = data.sell_price
                p.sell_reason = data.sell_reason
                p.profit_pct = parseFloat(((p.sell_price - p.buy_price) / p.buy_price * 100).toFixed(2))
                localStorage.setItem('guest_positions', JSON.stringify(positions))
                return { data: { message: '卖出已记录在本地', profit_pct: p.profit_pct } }
            }
            throw new Error('持仓未找到')
        }
        return api.post('/paper-trading/sell', data)
    },
    checkExit: async (code, buyPrice, buyDate, configParams = {}) => {
        const token = localStorage.getItem('quant_user_token')
        let params = { buy_price: buyPrice, buy_date: buyDate, ...configParams }

        if (!token || token === 'guest') {
            const localConfig = localStorage.getItem('quant_guest_config')
            if (localConfig) {
                try {
                    const conf = JSON.parse(localConfig)
                    params = { ...params, ...conf }
                } catch (e) { }
            }
        }
        return api.get(`/paper-trading/check-exit/${code}`, { params })
    },
    getHistory: async (limit = 50) => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            const history = positions.filter(p => p.status === 'closed').sort((a, b) => b.id - a.id).slice(0, limit)
            return { data: { trades: history } }
        }
        return api.get('/paper-trading/history', { params: { limit } })
    },
    toggleMonitoring: async (id) => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            const idx = positions.findIndex(p => p.id === id)
            if (idx !== -1) {
                positions[idx].monitoring = positions[idx].monitoring === 0 ? 1 : 0
                localStorage.setItem('guest_positions', JSON.stringify(positions))
                return { data: { monitoring: !!positions[idx].monitoring } }
            }
            throw new Error('未找到该持仓')
        }
        return api.post(`/paper-trading/toggle-monitoring/${id}`)
    },
    delete: async (id) => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            let positions = JSON.parse(localStorage.getItem('guest_positions') || '[]')
            positions = positions.filter(p => p.id !== id)
            localStorage.setItem('guest_positions', JSON.stringify(positions))
            return { data: { message: '本地记录已删除' } }
        }
        return api.delete(`/paper-trading/${id}`)
    }
}

// ── 数据中心 ────────────────────────────────────────
export const dataCenter = {
    getStatus: () => api.get('/data-center/status'),
    triggerUpdate: (params) => api.post('/data-center/update', params),
    getUpdateStatus: () => api.get('/data-center/update-status'),
}

// ── 技术分析 ────────────────────────────────────────
export const analysis = {
    getKline: (code, days = 250) => cachedGet(`/analysis/kline/${code}`, { params: { days } }),
    getTrendlines: (code, days = 250, long_period = null, short_period = null) =>
        cachedGet(`/analysis/trendlines/${code}`, { params: { days, long_period, short_period } }),
    getPatterns: (code, days = 100) => cachedGet(`/analysis/patterns/${code}`, { params: { days } }),
    getMarketStructure: (code, days = 250) => cachedGet(`/analysis/market-structure/${code}`, { params: { days } }),
}

// ── 配置中心 (系统全局) ──────────────────────────────────
export const configApi = {
    getFactorConfig: () => cachedGet('/config/factor'),
    updateFactorConfig: (updates) => {
        localStorage.removeItem(CACHE_PREFIX + '/config/factor{}')
        return api.put('/config/factor', { updates })
    },
    getStrategyConfig: () => cachedGet('/config/strategy'),
    updateStrategyConfig: (updates) => {
        localStorage.removeItem(CACHE_PREFIX + '/config/strategy{}')
        return api.put('/config/strategy', { updates })
    },
}

// ── 用户个人配置与鉴权 (支持游客本地) ─────────────────────────
export const authApi = {
    register: (data) => api.post('/auth/register', data),
    login: (data) => api.post('/auth/login', data),
    logout: () => api.post('/auth/logout'),
    getInfo: () => api.get('/auth/user/info'),
    getConfig: async () => {
        const token = localStorage.getItem('quant_user_token')
        if (!token || token === 'guest') {
            const localConfig = localStorage.getItem('quant_guest_config')
            return { data: { config_json: localConfig } }
        }
        return api.get('/auth/config')
    },
    saveConfig: async (config) => {
        const token = localStorage.getItem('quant_user_token')
        const config_json = typeof config === 'string' ? config : JSON.stringify(config)

        let result;
        if (!token || token === 'guest') {
            localStorage.setItem('quant_guest_config', config_json)
            result = { data: { message: '配置已保存至本地' } }
        } else {
            result = await api.post('/auth/config', { config_json })
        }

        // 清除相关缓存，确保配置更新后立即生效
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i)
            if (key && key.startsWith(CACHE_PREFIX)) {
                localStorage.removeItem(key)
                i-- // 调整索引
            }
        }
        return result
    }
}

export default api
