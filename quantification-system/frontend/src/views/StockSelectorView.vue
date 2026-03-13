<template>
  <div class="stock-selector page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">量化指挥中心</h1>
      <p class="page-subtitle">AI 驱动选股 · 实时因子分析 · 形态识别</p>
    </div>

    <!-- 操作栏 -->
    <div class="card mb-24">
      <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap">
        <div class="glass" style="padding: 4px 12px; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary)">模型</span>
          <el-select v-model="selectedModelSelection" placeholder="选择模型" size="default" style="width: 240px" clearable popper-class="dark-dropdown">
            <el-option-group v-for="m in models" :key="m.path" :label="m.name">
              <el-option v-if="m.types.includes('xgboost')" label="XGBoost 模型" :value="JSON.stringify({ path: m.path, types: ['xgboost'] })" />
              <el-option v-if="m.types.includes('lgbm')" label="LightGBM 模型" :value="JSON.stringify({ path: m.path, types: ['lgbm'] })" />
              <el-option v-if="m.types.length > 1" label="混合模型 (全部)" :value="JSON.stringify({ path: m.path, types: m.types })" />
            </el-option-group>
          </el-select>
        </div>

        <div class="glass" style="padding: 4px 12px; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary)">数量</span>
          <el-input-number v-model="topN" :min="5" :max="100" :step="5" size="default" style="width: 100px" />
        </div>

        <el-button type="primary" :loading="running" @click="runSelection" style="height: 42px; padding: 0 24px">
          <el-icon style="margin-right: 8px"><Search /></el-icon> 执行扫描
        </el-button>
        
        <el-button @click="refreshAll" :loading="loading" style="height: 42px">
          <el-icon><Refresh /></el-icon>
        </el-button>

        <div v-if="running" class="text-cyan" style="font-size: 13px; display: flex; align-items: center; gap: 10px">
          <el-icon class="loading-pulse"><Loading /></el-icon>
          <span>{{ taskStatus }}</span>
        </div>
      </div>
    </div>

    <!-- 结果表格 -->
    <div class="card" style="padding: 0">
      <div class="card-header" style="padding: 24px 24px 0 24px; margin-bottom: 16px">
        <span class="card-title">选股结果 <small class="text-muted" style="margin-left: 8px">找到 {{ items.length }} 只股票</small></span>
        <span class="text-mono" style="font-size: 11px; color: var(--text-muted)">{{ fileName ?? '' }}</span>
      </div>

      <el-tabs v-if="Object.keys(groupedItems).length > 0" v-model="activeModelTab" style="padding: 0 24px">
        <el-tab-pane v-for="(list, model) in groupedItems" :key="model" :label="model.toUpperCase()" :name="model" />
      </el-tabs>
      
      <el-table 
        :data="activeItems" 
        style="width: 100%" 
        max-height="650" 
        row-class-name="glass-row"
      >
        <el-table-column prop="stock_code" label="代码" width="160">
          <template #default="{ row }">
            <span class="text-mono" style="font-weight: 700; color: var(--accent-blue)">{{ String(row.stock_code).padStart(6, '0') }}</span>
            <span v-if="row.is_resonance" class="tag tag-buy" style="margin-left: 8px; font-size: 10px; padding: 2px 4px">共振</span>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="名称" width="120" />
        <el-table-column prop="signal" label="信号" width="100">
          <template #default="{ row }">
            <span :class="row.signal === 'buy' || row.signal === '买入' ? 'tag tag-buy' : 'tag tag-neutral'">
              {{ row.signal?.toUpperCase() ?? '—' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="胜率" width="130" sortable>
          <template #default="{ row }">
            <div style="display: flex; align-items: center; gap: 8px">
              <span class="text-mono" :style="{ color: row.confidence > 70 ? 'var(--accent-red)' : 'inherit' }">
                {{ row.confidence != null ? Number(row.confidence).toFixed(1) : '—' }}%
              </span>
              <div v-if="row.confidence" class="confidence-bar" :style="{ width: row.confidence + '%', background: row.confidence > 70 ? 'var(--accent-red)' : 'var(--accent-blue)' }"></div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="prediction" label="原始分数" width="120" sortable>
          <template #default="{ row }">
            <span class="text-muted text-mono">{{ row.prediction != null ? Number(row.prediction).toFixed(4) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="current_price" label="当前价格" width="100">
          <template #default="{ row }">
            <span class="text-mono" style="font-weight: 600">{{ row.current_price != null ? Number(row.current_price).toFixed(2) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="220" align="right">
          <template #default="{ row }">
            <div style="display: flex; gap: 8px; justify-content: flex-end; padding-right: 12px">
              <el-tooltip content="基本面分析" placement="top"><el-button size="small" circle type="warning" @click.stop="showFundamental(row.stock_code)"><el-icon><InfoFilled /></el-icon></el-button></el-tooltip>
              <el-tooltip content="因子快照" placement="top"><el-button size="small" circle @click.stop="showFactors(row.stock_code)"><el-icon><PieChart /></el-icon></el-button></el-tooltip>
              <el-tooltip content="形态识别" placement="top"><el-button size="small" circle @click.stop="showSignals(row.stock_code)"><el-icon><Lightning /></el-icon></el-button></el-tooltip>
              <el-tooltip content="技术分析" placement="top"><el-button size="small" circle type="primary" @click.stop="goAnalysis(row.stock_code)"><el-icon><TrendCharts /></el-icon></el-button></el-tooltip>
              <el-tooltip content="加入实盘验证" placement="top"><el-button size="small" circle type="success" style="background-color: var(--accent-green) !important; border-color: var(--accent-green) !important; color: #fff !important" @click.stop="addToPaperTrading(row)"><el-icon><Aim /></el-icon></el-button></el-tooltip>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 因子弹窗 -->
    <el-dialog v-model="factorVisible" title="因子快照" width="800px" custom-class="glass-dialog">
      <div v-if="factorLoading" class="flex-center" style="padding: 60px">
        <el-icon class="loading-pulse" :size="40" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else style="padding: 0 10px">
        <div class="mb-24" style="display: flex; justify-content: space-between; align-items: flex-end">
          <div>
            <h2 class="text-mono" style="font-size: 24px; color: var(--accent-blue)">{{ String(factorCode).padStart(6, '0') }}</h2>
            <p class="text-muted">基于近期市场数据计算的 {{ factorList.length }} 个表现因子</p>
          </div>
          <div class="tag tag-neutral">{{ factorData.latest_date ?? '最新' }}</div>
        </div>
        
        <div style="max-height: 500px; overflow-y: auto; padding-right: 8px">
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px">
            <div v-for="item in factorList.slice(0, 100)" :key="item.name" class="factor-item glass" :style="{ borderLeft: getFactorColor(item.name) }">
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
    <el-dialog v-model="signalVisible" title="形态识别" width="600px" custom-class="glass-dialog">
      <div v-if="signalLoading" class="flex-center" style="padding: 60px">
        <el-icon class="loading-pulse" :size="40" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px">
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
const factorList = computed(() => {
  if (factorData.value.factor_details) {
    return factorData.value.factor_details
  }
  return Object.entries(factorData.value.factors || {}).map(([name, value]) => ({ name, value }))
})

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

.factor-item {
  display: flex;
  justify-content: space-between;
  padding: 12px 16px;
  font-size: 13px;
  border-radius: var(--radius-md);
}

.factor-name {
  color: var(--text-secondary);
  font-weight: 500;
}

.factor-value {
  font-family: var(--font-mono);
  color: var(--accent-blue);
  font-weight: 600;
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

.stat-card.mini {
  padding: 12px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  text-align: center;
}

.stat-card.mini .stat-label {
  font-size: 11px;
  margin-bottom: 4px;
}

.stat-card.mini .stat-value.small {
  font-size: 18px;
  font-family: var(--font-mono);
  color: var(--accent-blue);
}

.mini-table :deep(.el-table__cell) {
  padding: 8px 0 !important;
}

.mini-table :deep(.el-table__header-wrapper th) {
  font-size: 11px;
  color: var(--text-muted);
}
</style>
