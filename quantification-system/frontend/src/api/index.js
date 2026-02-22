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

// ── 实盘验证 ────────────────────────────────────────
export const paperTrading = {
    getPositions: (status = 'active') => api.get('/paper-trading/positions', { params: { status } }),
    buy: (data) => api.post('/paper-trading/buy', data),
    sell: (data) => api.post('/paper-trading/sell', data),
    checkExit: (code, buyPrice, buyDate, configParams = {}) =>
        api.get(`/paper-trading/check-exit/${code}`, { params: { buy_price: buyPrice, buy_date: buyDate, ...configParams } }),
    getHistory: (limit = 50) => api.get('/paper-trading/history', { params: { limit } }),
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

// ── 配置中心 ────────────────────────────────────────
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

// ── 用户鉴权 ────────────────────────────────────────
export const authApi = {
    register: (data) => api.post('/auth/register', data),
    login: (data) => api.post('/auth/login', data),
    logout: () => api.post('/auth/logout'),
    getInfo: () => api.get('/auth/user/info'),
    getConfig: () => api.get('/auth/config'),
    saveConfig: (config_json) => api.post('/auth/config', { config_json })
}

export default api
