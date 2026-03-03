<template>
  <div class="paper-trading page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">实盘验证中心</h1>
      <p class="page-subtitle">人工交易监控 · 信号验证 · 绩效分析</p>
    </div>

    <!-- 统计 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">活跃监控</div>
        <div class="stat-value text-mono">{{ activePositions.length }}</div>
        <div class="stat-sub">当前实盘跟踪标的</div>
      </div>
      <div class="stat-card" style="border-bottom: 2px solid var(--accent-purple)">
        <div class="stat-label">已验证交易</div>
        <div class="stat-value text-mono">{{ closedCount }}</div>
        <div class="stat-sub">完成完整周期的交易数</div>
      </div>
      <div class="stat-card" :style="{ borderBottom: '2px solid ' + (winRate >= 50 ? 'var(--accent-red)' : 'var(--accent-green)') }">
        <div class="stat-label">胜率</div>
        <div class="stat-value text-mono" :class="winRate >= 50 ? 'text-up' : 'text-down'">{{ winRate.toFixed(1) }}%</div>
        <div class="stat-sub">实现正向收益的比例</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 24px">
      <!-- 买入表单 -->
      <div class="card" style="min-width: 0">
        <div class="card-header">
          <span class="card-title">新增持仓监控</span>
        </div>
        <el-form layout="vertical" :model="buyForm" @submit.prevent="handleBuy" style="overflow: hidden">
          <el-form-item label="股票信息">
            <div style="display: flex; gap: 10px; flex-wrap: wrap">
              <el-input v-model="buyForm.code" placeholder="代码" style="flex: 1; min-width: 100px" />
              <el-input v-model="buyForm.name" placeholder="名称" style="flex: 1.5; min-width: 120px" />
            </div>
          </el-form-item>
          <div style="display: flex; flex-direction: column; gap: 0">
            <el-form-item label="买入日期">
              <el-date-picker v-model="buyForm.buy_date" type="date" format="YYYY-MM-DD" value-format="YYYY-MM-DD" style="width: 100%" />
            </el-form-item>
            <el-form-item label="买入价格">
              <el-input-number v-model="buyForm.buy_price" :precision="3" :step="0.01" style="width: 100%" placeholder="待定" :value-on-clear="null" />
            </el-form-item>
          </div>
          <el-button type="primary" block @click="handleBuy" :loading="buying" style="width: 100%; height: 42px; margin-top: 10px">
            <el-icon style="margin-right: 8px"><Plus /></el-icon> 部署监控
          </el-button>
        </el-form>
      </div>

      <!-- 活跃持仓表 -->
      <div class="card" style="padding: 0">
        <div class="card-header" style="padding: 24px 24px 0 24px; margin-bottom: 16px">
          <span class="card-title">当前活跃持仓</span>
          <el-button circle size="small" @click="loadPositions" :loading="loading">
            <el-icon><Refresh /></el-icon>
          </el-button>
        </div>
        <el-table :data="activePositions" style="width: 100%" row-class-name="glass-row">
          <el-table-column prop="code" label="代码" width="100">
            <template #default="{ row }">
              <span class="text-mono" style="font-weight: 700; color: var(--accent-blue)">{{ String(row.code).padStart(6, '0') }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="name" label="名称" width="100" />
          <el-table-column prop="buy_price" label="买入价" width="100">
            <template #default="{ row }"><span class="text-mono">{{ Number(row.buy_price).toFixed(2) }}</span></template>
          </el-table-column>
          <el-table-column prop="latest_price" label="最新价" width="100">
            <template #default="{ row }"><span class="text-mono" style="font-weight: 600">{{ row.latest_price != null ? Number(row.latest_price).toFixed(2) : '—' }}</span></template>
          </el-table-column>
          <el-table-column prop="unrealized_pct" label="浮动盈亏" width="90" sortable>
            <template #default="{ row }">
              <span v-if="row.unrealized_pct != null" :class="row.unrealized_pct >= 0 ? 'text-up' : 'text-down'" class="text-mono" style="font-weight: 800">
                {{ row.unrealized_pct > 0 ? '+' : '' }}{{ row.unrealized_pct }}%
              </span>
              <span v-else class="text-muted">—</span>
            </template>
          </el-table-column>
          <el-table-column label="智能分析" min-width="280">
            <template #default="{ row }">
              <div v-if="analysisMap[row.id]" style="padding: 10px 0">
                <div v-if="analysisMap[row.id].status === 'pending'" class="text-secondary" style="font-size: 12px">
                  <el-icon class="is-loading"><Loading /></el-icon> {{ analysisMap[row.id].message }}
                </div>
                <div v-else>
                  <div v-for="(cond, key) in analysisMap[row.id].conditions" :key="key" style="margin-bottom: 6px">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px">
                      <span style="font-size: 11px; font-weight: 600">{{ cond.label }}</span>
                      <span v-if="cond.price" style="font-size: 10px; color: var(--text-secondary)">
                        {{ Number(row.latest_price).toFixed(2) }}<!--
                        --><span style="color: var(--text-muted); margin: 0 4px">/</span><!--
                        -->{{ Number(cond.price).toFixed(2) }}
                      </span>
                      <span v-else-if="key === 'time_stop'" style="font-size: 10px; color: var(--text-secondary)">
                        {{ cond.holding_days }}<span style="color: var(--text-muted); margin: 0 4px">/</span>{{ cond.max_days }} 天
                      </span>
                      <span v-if="cond.triggered" class="text-up" style="font-size: 10px; font-weight: 800">已触发</span>
                      <span v-else style="font-size: 10px; color: var(--text-muted)">{{ cond.progress }}%</span>
                    </div>
                    <el-progress 
                      :percentage="cond.progress" 
                      :status="cond.triggered ? 'exception' : (cond.progress > 80 ? 'warning' : '')" 
                      :show-text="false" 
                      :stroke-width="4"
                    />
                  </div>
                  <span v-if="!hasTriggered(row.id) && Object.keys(analysisMap[row.id].conditions).length > 0" class="text-muted" style="font-size: 10px">
                    <el-icon color="var(--accent-green)"><SuccessFilled /></el-icon> 安全持有中
                  </span>
                  <span v-if="Object.keys(analysisMap[row.id].conditions).length === 0" class="text-muted" style="font-size: 10px">未启用智能监控</span>
                </div>
              </div>
              <div v-else-if="row.buy_price != null">
                <el-icon class="is-loading" color="var(--accent-blue)"><Loading /></el-icon>
              </div>
              <div v-else>
                <span class="text-muted" style="font-size: 11px">等待价格补充</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="操作" align="right" width="150" fixed="right">
            <template #default="{ row }">
              <div style="display: flex; gap: 8px; justify-content: flex-end; padding-right: 4px">
                <el-tooltip content="了结获利/止损" placement="top"><el-button size="small" circle type="danger" @click="openSellDialog(row)"><el-icon><Sell /></el-icon></el-button></el-tooltip>
                <el-tooltip content="技术分析" placement="top"><el-button size="small" circle @click="$router.push({ path: '/analysis', query: { code: row.code } })"><el-icon><TrendCharts /></el-icon></el-button></el-tooltip>
                <el-tooltip content="删除监控" placement="top"><el-button size="small" circle type="info" plain @click="handleDelete(row)"><el-icon><Delete /></el-icon></el-button></el-tooltip>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- 历史交易 -->
    <div class="card mt-24" style="padding: 0; margin-top: 24px">
      <div class="card-header" style="padding: 24px 24px 0 24px; margin-bottom: 16px">
        <span class="card-title">历史交易记录</span>
        <el-button circle size="small" @click="loadHistory" :loading="historyLoading">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
      <el-table :data="historyTrades" style="width: 100%" max-height="400">
        <el-table-column prop="code" label="代码" width="100">
          <template #default="{ row }">{{ String(row.code).padStart(6, '0') }}</template>
        </el-table-column>
        <el-table-column prop="name" label="名称" width="100" />
        <el-table-column prop="buy_date" label="买入日" width="110" />
        <el-table-column prop="sell_date" label="卖出日" width="110" />
        <el-table-column prop="profit_pct" label="净盈亏" width="120" sortable>
          <template #default="{ row }">
            <span :class="row.profit_pct >= 0 ? 'text-up' : 'text-down'" class="text-mono" style="font-weight: 800">
              {{ row.profit_pct > 0 ? '+' : '' }}{{ row.profit_pct }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="sell_reason" label="退出原因" />
      </el-table>
    </div>


    <!-- 卖出弹窗 -->
    <el-dialog v-model="sellVisible" title="了结交易" width="400px" custom-class="glass-dialog">
      <el-form :model="sellForm" label-position="top">
        <el-form-item label="卖出日期">
          <el-date-picker v-model="sellForm.sell_date" type="date" format="YYYY-MM-DD" value-format="YYYY-MM-DD" style="width: 100%" />
        </el-form-item>
        <el-form-item label="卖出价格">
          <el-input-number v-model="sellForm.sell_price" :precision="3" :step="0.01" style="width: 100%" />
        </el-form-item>
        <el-form-item label="卖出原因 / 备注">
          <el-input v-model="sellForm.sell_reason" placeholder="例: 触发止损" />
        </el-form-item>
      </el-form>
      <template #footer>
        <div style="display: flex; gap: 12px">
          <el-button @click="sellVisible = false" style="flex: 1">取消</el-button>
          <el-button type="danger" @click="handleSell" :loading="selling" style="flex: 1">确认卖出</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { paperTrading, authApi } from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, Loading, TrendCharts, Aim, Sell, Delete, SuccessFilled, InfoFilled } from '@element-plus/icons-vue'

const loading = ref(false)
const buying = ref(false)
const selling = ref(false)
const historyLoading = ref(false)
const activePositions = ref([])
const historyTrades = ref([])
const closedCount = ref(0)

const winRate = computed(() => {
  if (!historyTrades.value.length) return 0
  const wins = historyTrades.value.filter(t => t.profit_pct > 0).length
  return (wins / historyTrades.value.length) * 100
})

// 买入表单
const buyForm = ref({ code: '', name: '', buy_date: '', buy_price: null, quantity: 1 })

const userConfig = ref(null)
const analysisMap = ref({}) // { position_id: exitData }

function hasTriggered(id) {
  if (!analysisMap.value[id]) return false
  return Object.values(analysisMap.value[id].conditions || {}).some(c => c.triggered)
}

// 简单的字符串哈希函数，用于生成配置指纹
function getHash(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i)
    hash |= 0
  }
  return Math.abs(hash).toString(36)
}

function getTodayStr() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`
}

// 卖出弹窗
const sellVisible = ref(false)
const sellForm = ref({ position_id: 0, sell_date: '', sell_price: 0, sell_reason: '手动卖出' })

onMounted(async () => { 
  // 加载配置，确保分析逻辑正确
  try {
    const res = await authApi.getConfig()
    if (res.data && res.data.config_json) {
      userConfig.value = typeof res.data.config_json === 'string' 
        ? JSON.parse(res.data.config_json) 
        : res.data.config_json
    }
  } catch (e) { console.error('Load config failed', e) }
  
  loadPositions(); 
  loadHistory() 
})

async function loadPositions() {
  loading.value = true
  try {
    const { data } = await paperTrading.getPositions('active')
    activePositions.value = data.positions
        
    // Wait for auto analysis silently
    setTimeout(() => {
        performAutoAnalysis(activePositions.value)
    }, 500)
    
  } catch (e) {
    ElMessage.error('加载持仓失败')
  } finally {
    loading.value = false
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const { data } = await paperTrading.getHistory(100)
    historyTrades.value = data.trades
    closedCount.value = data.trades.length
  } catch (e) {
    ElMessage.error('加载历史交易失败')
  } finally {
    historyLoading.value = false
  }
}

async function handleBuy() {
  if (!buyForm.value.code || !buyForm.value.buy_date) {
    ElMessage.warning('代码与日期是必填项')
    return
  }
  buying.value = true
  try {
    await paperTrading.buy(buyForm.value)
    ElMessage.success('成功部署监控')
    buyForm.value = { code: '', name: '', buy_date: '', buy_price: null, quantity: 1 }
    loadPositions()
    loadHistory()
  } catch (e) {
    console.error(e)
    ElMessage.error('记录买入失败')
  } finally {
    buying.value = false
  }
}

async function performAutoAnalysis(positions) {
  const today = getTodayStr()
  for (const pos of positions) {
    if (pos.buy_price == null) continue
    
    // Check Cache (Include config hash to bust cache if settings change)
    // 使用完整序列化后的哈希，确保任何微小改动都能触发刷新
    const configHash = userConfig.value ? getHash(JSON.stringify(userConfig.value)) : 'default'
    const cacheKey = `quant_analysis_v7_${pos.code}_${pos.buy_date}_${pos.buy_price}_${today}_${configHash}`
    const cached = localStorage.getItem(cacheKey)
    if (cached) {
      try {
        const parsed = JSON.parse(cached)
        // If it was pending, we might want to re-fetch
        if (parsed.status !== 'pending') {
            analysisMap.value[pos.id] = parsed
            continue
        }
      } catch (e) {}
    }
    
    // Fetch
    try {
      const configParams = userConfig.value || {}
      const { data } = await paperTrading.checkExit(pos.code, pos.buy_price, pos.buy_date, configParams)
      analysisMap.value[pos.id] = data
      if (data.status !== 'pending') {
          localStorage.setItem(cacheKey, JSON.stringify(data))
      }
    } catch(e) {
      console.error('Failed to auto-analyze', pos.code, e)
    }
  }
}

async function handleDelete(row) {
    try {
        await ElMessageBox.confirm('确定要删除这条监控记录吗？', '确认删除', { type: 'warning' })
        await paperTrading.delete(row.id)
        ElMessage.success('已删除记录')
        loadPositions()
    } catch (e) {
        if (e !== 'cancel') ElMessage.error('删除操作失败')
    }
}



function openSellDialog(row) {
  sellForm.value = { position_id: row.id, sell_date: '', sell_price: row.latest_price ?? 0, sell_reason: '策略退出' }
  sellVisible.value = true
}

async function handleSell() {
  if (!sellForm.value.sell_price || !sellForm.value.sell_date) {
    ElMessage.warning('请输入价格及日期')
    return
  }
  selling.value = true
  try {
    const { data } = await paperTrading.sell(sellForm.value)
    ElMessage.success(`交易已了结。净盈亏: ${data.profit_pct}%`)
    sellVisible.value = false
    loadPositions()
    loadHistory()
  } catch (e) {
    ElMessage.error('验证失败')
  } finally {
    selling.value = false
  }
}
</script>

<style scoped>
.exit-condition-item {
  padding: 16px;
  margin-bottom: 12px;
  transition: all 0.3s;
  border-left: 4px solid var(--text-muted);
}
.exit-condition-item.triggered {
  border-left-color: var(--accent-red);
  background: rgba(255, 51, 102, 0.05);
}
.exit-condition-item .label {
  font-weight: 700;
  font-size: 14px;
}
.exit-condition-item .details {
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  display: flex;
  gap: 16px;
}
</style>
