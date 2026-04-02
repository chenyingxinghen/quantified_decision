<template>
  <div class="stock-selector page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">量化指挥中心</h1>
      <p class="page-subtitle">AI 驱动选股 · 实时因子分析 · 形态识别</p>
    </div>

    <!-- 操作栏 -->
    <div class="card mb-24">
      <div class="flex-wrap-mobile" style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap">
        <div class="glass w-full-mobile" style="padding: 4px 12px; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary)">模型</span>
          <el-select v-model="selectedModelSelection" placeholder="选择模型" size="default" style="flex: 1; min-width: 140px" clearable popper-class="dark-dropdown">
            <el-option-group v-for="m in models" :key="m.path" :label="m.name">
              <el-option v-if="m.types.includes('xgboost')" label="XGBoost 模型" :value="JSON.stringify({ path: m.path, types: ['xgboost'] })" />
              <el-option v-if="m.types.includes('lgbm')" label="LightGBM 模型" :value="JSON.stringify({ path: m.path, types: ['lgbm'] })" />
              <el-option v-if="m.types.length > 1" label="混合模型 (全部)" :value="JSON.stringify({ path: m.path, types: m.types })" />
            </el-option-group>
          </el-select>
        </div>

        <div class="glass w-full-mobile" style="padding: 4px 12px; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary)">数量</span>
          <el-input-number v-model="topN" :min="5" :max="100" :step="5" size="default" style="flex: 1" />
        </div>

        <div class="flex-center gap-12 w-full-mobile" style="justify-content: flex-start">
          <el-button type="primary" :loading="running" @click="runSelection" style="height: 40px; flex: 1; padding: 0 16px">
            <el-icon style="margin-right: 8px"><Search /></el-icon> 执行扫描
          </el-button>
          
          <el-button @click="refreshAll" :loading="loading" style="height: 40px; width: 44px">
            <el-icon><Refresh /></el-icon>
          </el-button>
        </div>

        <div v-if="running" class="text-cyan" style="font-size: 13px; display: flex; align-items: center; gap: 10px">
          <el-icon class="loading-pulse"><Loading /></el-icon>
          <span>{{ taskStatus }}</span>
        </div>
      </div>
    </div>

    <!-- 结果表格 / 移动端卡片列表 -->
    <div class="card" style="padding: 0">
      <div class="card-header" style="padding: 24px 24px 0 24px; margin-bottom: 16px">
        <span class="card-title">选股结果 <small class="text-muted" style="margin-left: 8px">找到 {{ items.length }} 只股票</small></span>
        <span class="text-mono" style="font-size: 11px; color: var(--text-muted)">{{ fileName ?? '' }}</span>
      </div>

      <el-tabs v-if="Object.keys(groupedItems).length > 0" v-model="activeModelTab" style="padding: 0 24px">
        <el-tab-pane v-for="(list, model) in groupedItems" :key="model" :label="model.toUpperCase()" :name="model" />
      </el-tabs>

      <!-- 桌面端表格 -->
      <el-table 
        :data="activeItems" 
        style="width: 100%" 
        max-height="650" 
        row-class-name="glass-row"
        class="selector-table desktop-table"
      >
        <el-table-column prop="stock_code" label="代码" width="90">
          <template #default="{ row }">
            <span class="text-mono" style="font-weight: 700; color: var(--accent-blue)">{{ String(row.stock_code).padStart(6, '0') }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="名称" min-width="80">
          <template #default="{ row }">
            <div style="display: flex; flex-direction: column; align-items: flex-start">
              <span>{{ row.name }}</span>
              <span v-if="row.is_resonance" class="tag tag-buy" style="font-size: 9px; padding: 0px 4px; transform: scale(0.9); transform-origin: left">共振</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="胜率" min-width="110" sortable>
          <template #default="{ row }">
            <div style="display: flex; align-items: center; gap: 8px">
              <span class="text-mono" :style="{ color: row.confidence > 70 ? 'var(--accent-red)' : 'inherit' }">
                {{ row.confidence != null ? Number(row.confidence).toFixed(1) : '—' }}%
              </span>
              <div v-if="row.confidence" class="confidence-bar" :style="{ width: row.confidence + '%', background: row.confidence > 70 ? 'var(--accent-red)' : 'var(--accent-blue)' }"></div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="current_price" label="价格" min-width="80">
          <template #default="{ row }">
            <span class="text-mono" style="font-weight: 600">{{ row.current_price != null ? Number(row.current_price).toFixed(2) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="prediction" label="分数" min-width="90" sortable>
          <template #default="{ row }">
            <span class="text-muted text-mono">{{ row.prediction != null ? Number(row.prediction).toFixed(4) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="180" align="right" fixed="right">
          <template #default="{ row }">
            <div class="action-btns">
              <el-tooltip content="基本面" placement="top"><el-button size="small" circle type="warning" @click.stop="showFundamental(row.stock_code)"><el-icon><InfoFilled /></el-icon></el-button></el-tooltip>
              <el-tooltip content="因子" placement="top"><el-button size="small" circle @click.stop="showFactors(row.stock_code)"><el-icon><PieChart /></el-icon></el-button></el-tooltip>
              <el-tooltip content="形态" placement="top"><el-button size="small" circle @click.stop="showSignals(row.stock_code)"><el-icon><Lightning /></el-icon></el-button></el-tooltip>
              <el-tooltip content="K线" placement="top"><el-button size="small" circle type="primary" @click.stop="goAnalysis(row.stock_code)"><el-icon><TrendCharts /></el-icon></el-button></el-tooltip>
              <el-tooltip content="加入实盘" placement="top"><el-button size="small" circle type="success" style="background-color: var(--accent-green) !important; border-color: var(--accent-green) !important; color: #fff !important" @click.stop="addToPaperTrading(row)"><el-icon><Aim /></el-icon></el-button></el-tooltip>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 移动端卡片列表 -->
      <div class="mobile-stock-list">
        <div v-for="row in activeItems" :key="row.stock_code" class="mobile-stock-card glass">
          <div class="msc-top">
            <div class="msc-info">
              <span class="text-mono msc-code">{{ String(row.stock_code).padStart(6, '0') }}</span>
              <span class="msc-name">{{ row.name }}</span>
              <span v-if="row.is_resonance" class="tag tag-buy" style="font-size: 9px; padding: 0 4px">共振</span>
            </div>
            <div class="msc-confidence">
              <span class="text-mono" :style="{ color: row.confidence > 70 ? 'var(--accent-red)' : 'var(--text-primary)', fontWeight: 700, fontSize: '16px' }">
                {{ row.confidence != null ? Number(row.confidence).toFixed(1) : '—' }}%
              </span>
              <div class="msc-bar-wrap">
                <div class="msc-bar" :style="{ width: (row.confidence || 0) + '%', background: row.confidence > 70 ? 'var(--accent-red)' : 'var(--accent-blue)' }"></div>
              </div>
            </div>
          </div>
          <div class="msc-meta">
            <span class="msc-meta-item">价格 <b>{{ row.current_price != null ? Number(row.current_price).toFixed(2) : '—' }}</b></span>
            <span class="msc-meta-item">分数 <b class="text-muted">{{ row.prediction != null ? Number(row.prediction).toFixed(4) : '—' }}</b></span>
          </div>
          <div class="msc-actions">
            <el-button size="small" type="warning" @click.stop="showFundamental(row.stock_code)"><el-icon><InfoFilled /></el-icon> 基本面</el-button>
            <el-button size="small" @click.stop="showFactors(row.stock_code)"><el-icon><PieChart /></el-icon> 因子</el-button>
            <el-button size="small" @click.stop="showSignals(row.stock_code)"><el-icon><Lightning /></el-icon> 形态</el-button>
            <el-button size="small" type="primary" @click.stop="goAnalysis(row.stock_code)"><el-icon><TrendCharts /></el-icon> K线</el-button>
            <el-button size="small" type="success" style="background-color: var(--accent-green) !important; border-color: var(--accent-green) !important; color: #fff !important" @click.stop="addToPaperTrading(row)"><el-icon><Aim /></el-icon> 实盘</el-button>
          </div>
        </div>
        <div v-if="activeItems.length === 0" class="text-muted" style="padding: 40px; text-align: center">暂无数据</div>
      </div>
    </div>

    <!-- 因子弹窗 -->
    <el-dialog v-model="factorVisible" title="因子快照" :width="isMobile ? '95%' : '800px'" custom-class="glass-dialog">
      <div v-if="factorLoading" class="flex-center" style="padding: 60px">
        <el-icon class="loading-pulse" :size="40" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else style="padding: 0 4px">
        <div class="mb-16" style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px">
          <div>
            <h2 class="text-mono" style="font-size: 22px; color: var(--accent-blue)">{{ String(factorCode).padStart(6, '0') }}</h2>
            <p class="text-muted" style="font-size: 12px">{{ factorList.length }} 个因子</p>
          </div>
          <div class="tag tag-neutral">{{ factorData.latest_date ?? '最新' }}</div>
        </div>

        <!-- 因子分组 tabs -->
        <el-tabs v-model="factorTab" size="small" style="margin-bottom: 8px">
          <el-tab-pane label="基本面" name="fundamental" />
          <el-tab-pane label="量价" name="price_vol" />
          <el-tab-pane label="估值" name="valuation" />
          <el-tab-pane label="全部" name="all" />
        </el-tabs>
        
        <div style="max-height: 420px; overflow-y: auto; padding-right: 4px">
          <div class="factor-grid">
            <div v-for="item in filteredFactorList" :key="item.name" class="factor-item glass" :style="{ borderLeft: getFactorColor(item.name) }">
              <el-tooltip :content="item.name" placement="top">
                <span class="factor-name">{{ formatFactorName(item.name) }}</span>
              </el-tooltip>
              <span class="factor-value">{{ formatFactorValue(item.value) }}</span>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- 信号弹窗 -->
    <el-dialog v-model="signalVisible" title="形态识别" :width="isMobile ? '95%' : '600px'" custom-class="glass-dialog">
      <div v-if="signalLoading" class="flex-center" style="padding: 60px">
        <el-icon class="loading-pulse" :size="40" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else>
        <div class="grid-2-mobile-1" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px">
          <div class="stat-card" style="border-bottom: 2px solid var(--accent-red)">
            <div class="stat-label">看多强度</div>
            <div class="stat-value text-up">{{ signalData.bullish_score ?? 0 }}</div>
          </div>
          <div class="stat-card" style="border-bottom: 2px solid var(--accent-green)">
            <div class="stat-label">看空强度</div>
            <div class="stat-value text-down">{{ signalData.bearish_score ?? 0 }}</div>
          </div>
        </div>
        
        <div class="mb-24">
          <h4 class="mb-16" style="color: var(--accent-red); display: flex; align-items: center; gap: 8px">
            <el-icon><CaretTop /></el-icon> 看多形态
          </h4>
          <div v-for="p in signalData.bullish_patterns" :key="p.description" class="pattern-badge bullish">
            <strong>{{ p.description }}</strong>
            <span class="score">+{{ p.score }}</span>
          </div>
          <div v-if="!signalData.bullish_patterns?.length" class="text-muted" style="padding-left: 12px">未检测到显著看多形态</div>
        </div>

        <div>
          <h4 class="mb-16" style="color: var(--accent-green); display: flex; align-items: center; gap: 8px">
            <el-icon><CaretBottom /></el-icon> 看空形态
          </h4>
          <div v-for="p in signalData.bearish_patterns" :key="p.description" class="pattern-badge bearish">
            <strong>{{ p.description }}</strong>
            <span class="score">-{{ p.score }}</span>
          </div>
          <div v-if="!signalData.bearish_patterns?.length" class="text-muted" style="padding-left: 12px">未检测到显著看空形态</div>
        </div>
      </div>
    </el-dialog>


  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { stockSelector } from '../api'
import { ElMessage } from 'element-plus'
import { 
  Search, Refresh, Loading, PieChart, InfoFilled,
  TrendCharts, Lightning, CaretTop, CaretBottom, Aim
} from '@element-plus/icons-vue'
import { paperTrading } from '../api'

const router = useRouter()
const loading = ref(false)
const running = ref(false)
const taskStatus = ref('')
const topN = ref(20)
const items = ref([])
const fileName = ref(null)

const models = ref([])
const selectedModelSelection = ref(null)

const activeModelTab = ref('')
const groupedItems = computed(() => {
  const groups = {}
  items.value.forEach(item => {
    const type = item.model_type || '已选'
    if (!groups[type]) groups[type] = []
    groups[type].push(item)
  })
  return groups
})
const activeItems = computed(() => {
  if (!activeModelTab.value && Object.keys(groupedItems.value).length > 0) {
    activeModelTab.value = Object.keys(groupedItems.value)[0]
  }
  return groupedItems.value[activeModelTab.value] || []
})

const factorVisible = ref(false)
const factorLoading = ref(false)
const factorCode = ref('')
const factorData = ref({})
const factorTab = ref('all')
const factorList = computed(() => {
  if (factorData.value.factor_details) {
    return factorData.value.factor_details
  }
  return Object.entries(factorData.value.factors || {}).map(([name, value]) => ({ name, value }))
})

const filteredFactorList = computed(() => {
  const list = factorList.value.slice(0, 100)
  if (factorTab.value === 'all') return list
  return list.filter(item => {
    const n = item.name.toLowerCase()
    if (factorTab.value === 'fundamental') return n.includes('roe') || n.includes('rev') || n.includes('np') || n.includes('xsjll') || n.includes('zzcjll') || n.includes('zcfzl') || n.includes('ocf')
    if (factorTab.value === 'price_vol') return n.includes('pv_') || n.includes('vol') || n.includes('sync') || n.includes('greed') || n.includes('buy') || n.includes('amount') || n.includes('turnover')
    if (factorTab.value === 'valuation') return n.includes('pe') || n.includes('pb') || n.includes('peg')
    return true
  })
})

const isMobile = computed(() => window.innerWidth <= 768)

const signalVisible = ref(false)
const signalLoading = ref(false)
const signalData = ref({})



let pollTimer = null

onMounted(() => {
  loadLatest()
  loadModels()
  checkTaskStatus()
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

async function refreshAll() {
  await Promise.all([loadModels(), loadLatest()])
}

async function loadModels() {
  try {
    const { data } = await stockSelector.getModels()
    models.value = data.models.filter(m => m.path.includes('mark') || m.path.includes('models/mark') || m.path.includes('models\\\\mark')) || []
  } catch (e) {
    console.error('加载模型失败', e)
  }
}

async function loadLatest() {
  loading.value = true
  try {
    const { data } = await stockSelector.getLatest()
    items.value = data.items ?? []
    fileName.value = data.file
    if (Object.keys(groupedItems.value).length > 0) {
      activeModelTab.value = Object.keys(groupedItems.value)[0]
    }
  } catch (e) {
    ElMessage.error('加载最新结果失败')
  } finally {
    loading.value = false
  }
}

async function runSelection() {
  if (running.value) return
  let modelParams = {}
  if (selectedModelSelection.value) {
    try {
      const parsed = JSON.parse(selectedModelSelection.value)
      modelParams.model_path = parsed.path
      modelParams.model_types = parsed.types
    } catch (e) { console.error('解析错误', e) }
  }

  // 将游客本地配置指纹传给后端，以便基础筛选条件生效
  const token = localStorage.getItem('quant_user_token')
  if (!token || token === 'guest') {
    const guestConfig = localStorage.getItem('quant_guest_config')
    if (guestConfig) modelParams.guest_config = guestConfig
  }

  try {
    await stockSelector.runSelection({ 
      top_n: topN.value,
      ...modelParams
    })
    running.value = true
    taskStatus.value = '正在初始化扫描...'
    startPolling()
  } catch (e) {
    ElMessage.error('选股任务启动失败')
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(checkTaskStatus, 2000)
}

async function checkTaskStatus() {
  try {
    const { data } = await stockSelector.getSelectionStatus()
    if (data.running) {
      running.value = true
      taskStatus.value = data.progress
      if (!pollTimer) startPolling()
    } else {
      if (running.value && !data.error) {
        ElMessage.success('选股扫描成功完成')
        items.value = data.items ?? []
        fileName.value = data.file
        if (Object.keys(groupedItems.value).length > 0) {
          activeModelTab.value = Object.keys(groupedItems.value)[0]
        }
      } else if (data.error) {
        ElMessage.error('选股失败: ' + data.error)
      }
      running.value = false
      if (pollTimer) {
        clearInterval(pollTimer)
        pollTimer = null
      }
    }
  } catch (e) {
    console.error('轮询错误', e)
  }
}

async function showFactors(code) {
  factorCode.value = code
  factorVisible.value = true
  factorLoading.value = true
  try {
    const { data } = await stockSelector.getFactors(code)
    factorData.value = data
  } catch (e) {
    ElMessage.error('获取因子数据失败')
  } finally {
    factorLoading.value = false
  }
}

async function showSignals(code) {
  signalVisible.value = true
  signalLoading.value = true
  try {
    const { data } = await stockSelector.getSignals(code)
    signalData.value = data
  } catch (e) {
    ElMessage.error('获取信号数据失败')
  } finally {
    signalLoading.value = false
  }
}

function showFundamental(code) {
  router.push({ path: '/fundamental', query: { code } })
}

function goAnalysis(code) {
  router.push({ path: '/analysis', query: { code } })
}

function getTodayStr() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

async function addToPaperTrading(row) {
  try {
    // 选股日通常是结果文件中的日期，如果没有则用今天
    const selectionDate = row.date || fileName.value?.match(/\d{4}-\d{2}-\d{2}/)?.[0] || getTodayStr()
    
    // 计算下一日（这里简单加1天，后续后端会自动匹配开盘价）
    const d = new Date(selectionDate)
    d.setDate(d.getDate() + 1)
    const nextDay = d.toISOString().split('T')[0]

    await paperTrading.buy({
      code: row.stock_code,
      name: row.name,
      buy_date: nextDay,
      buy_price: null, // 标记为下个交易日开盘价待定
      quantity: 1
    })
    ElMessage.success(`${row.name} 已加入实盘验证中心 (待定下一日开盘价)`)
  } catch (e) {
    console.error(e)
    ElMessage.error('加入实盘验证失败')
  }
}

function formatFactorName(name) {
  // 简单翻译/美化常见因子名
  const dict = {
    'roe_jq': 'ROE(加权)', 'roe_kc': 'ROE(扣非)', 'xsjll': '销售净利率', 'zzcjll': '总资产收益率',
    'rev_yoy': '营收同比', 'np_yoy': '净利润同比', 'np_kc_yoy': '扣非净利同比',
    'dynamic_pe': '动态PE', 'dynamic_pb': '动态PB', 'peg': 'PEG',
    'zcfzl': '资产负债率', 'qycs': '权益乘数', 'ocf_to_eps': '现金流/EPS',
    'pv_sync_5': '量价协同(5)', 'greed_index': '贪婪指数', 'turnover_zscore': '换手率Z分',
    'buy_vol_ratio_10': '买盘比', 'net_buy_ratio_10': '净买比', 'amount_accel': '成交额加速'
  }
  return dict[name] || name
}

function formatFactorValue(val) {
  if (val == null) return 'N/A'
  const v = Number(val)
  if (isNaN(v)) return val
  if (Math.abs(v) > 1000) return v.toFixed(0)
  if (Math.abs(v) > 100) return v.toFixed(1)
  return v.toFixed(4)
}

function getFactorColor(name) {
  const n = name.toLowerCase()
  if (n.includes('roe') || n.includes('rev') || n.includes('np')) return '3px solid var(--accent-red)'
  if (n.includes('pv_') || n.includes('vol') || n.includes('sync') || n.includes('greed')) return '3px solid var(--accent-blue)'
  if (n.includes('pe') || n.includes('pb') || n.includes('peg')) return '3px solid var(--accent-green)'
  return '3px solid transparent'
}
</script>

<style scoped>
.confidence-bar {
  height: 4px;
  border-radius: 2px;
  max-width: 60px;
}

.action-btns {
  display: flex;
  gap: 6px;
  justify-content: flex-end;
  padding-right: 8px;
  flex-wrap: wrap;
}

/* 桌面端显示表格，隐藏卡片 */
.desktop-table { display: table; }
.mobile-stock-list { display: none; }

@media (max-width: 768px) {
  .desktop-table { display: none !important; }
  .mobile-stock-list { display: block; padding: 0 12px 16px; }
}

/* 移动端股票卡片 */
.mobile-stock-card {
  border-radius: var(--radius-md);
  padding: 14px 16px;
  margin-bottom: 10px;
  border: 1px solid var(--border-color);
}

.msc-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
}

.msc-info {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.msc-code {
  font-weight: 700;
  color: var(--accent-blue);
  font-size: 15px;
}

.msc-name {
  font-size: 14px;
  font-weight: 600;
}

.msc-confidence {
  text-align: right;
  min-width: 60px;
}

.msc-bar-wrap {
  width: 60px;
  height: 4px;
  background: rgba(255,255,255,0.08);
  border-radius: 2px;
  margin-top: 4px;
  margin-left: auto;
}

.msc-bar {
  height: 100%;
  border-radius: 2px;
  max-width: 100%;
}

.msc-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 12px;
}

.msc-meta-item b {
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.msc-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.msc-actions .el-button {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  padding: 6px 4px;
}

/* 因子网格 */
.factor-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

@media (max-width: 768px) {
  .factor-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 400px) {
  .factor-grid {
    grid-template-columns: 1fr;
  }
}

.factor-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  font-size: 12px;
  border-radius: var(--radius-md);
  gap: 6px;
}

.factor-name {
  color: var(--text-secondary);
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 70%;
}

.factor-value {
  font-family: var(--font-mono);
  color: var(--accent-blue);
  font-weight: 600;
  white-space: nowrap;
}

.pattern-badge {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  border-radius: var(--radius-md);
  margin-bottom: 12px;
  font-size: 14px;
}

.pattern-badge.bullish {
  background: rgba(255, 51, 102, 0.08);
  border: 1px solid rgba(255, 51, 102, 0.15);
}

.pattern-badge.bearish {
  background: rgba(0, 255, 136, 0.08);
  border: 1px solid rgba(0, 255, 136, 0.15);
}

.pattern-badge .score {
  font-family: var(--font-mono);
  font-weight: 800;
}
</style>
