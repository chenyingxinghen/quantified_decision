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
      <div class="card">
        <div class="card-header">
          <span class="card-title">新增持仓监控</span>
        </div>
        <el-form layout="vertical" :model="buyForm" @submit.prevent="handleBuy">
          <el-form-item label="标的代码及名称">
            <div style="display: flex; gap: 10px">
              <el-input v-model="buyForm.code" placeholder="代码 (如 600000)" />
              <el-input v-model="buyForm.name" placeholder="名称" />
            </div>
          </el-form-item>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px">
            <el-form-item label="买入日期">
              <el-date-picker v-model="buyForm.buy_date" type="date" format="YYYY-MM-DD" value-format="YYYY-MM-DD" style="width: 100%" />
            </el-form-item>
            <el-form-item label="买入价格">
              <el-input-number v-model="buyForm.buy_price" :precision="3" :step="0.01" style="width: 100%" />
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
          <el-table-column prop="code" label="代码" width="110">
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
          <el-table-column prop="unrealized_pct" label="浮动盈亏" width="110" sortable>
            <template #default="{ row }">
              <span v-if="row.unrealized_pct != null" :class="row.unrealized_pct >= 0 ? 'text-up' : 'text-down'" class="text-mono" style="font-weight: 800">
                {{ row.unrealized_pct > 0 ? '+' : '' }}{{ row.unrealized_pct }}%
              </span>
              <span v-else class="text-muted">—</span>
            </template>
          </el-table-column>
          <el-table-column label="操作" align="right" min-width="180">
            <template #default="{ row }">
              <div style="display: flex; gap: 8px; justify-content: flex-end; padding-right: 12px">
                <el-button size="small" circle @click="checkExit(row)"><el-icon><Aim /></el-icon></el-button>
                <el-button size="small" circle type="danger" @click="openSellDialog(row)"><el-icon><Sell /></el-icon></el-button>
                <el-button size="small" circle @click="$router.push({ path: '/analysis', query: { code: row.code } })"><el-icon><TrendCharts /></el-icon></el-button>
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

    <!-- 卖出条件检测弹窗 -->
    <el-dialog v-model="exitVisible" title="智能退出分析" width="550px" custom-class="glass-dialog">
      <div v-if="exitLoading" class="flex-center" style="padding: 60px">
        <el-icon class="loading-pulse" :size="40" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else-if="exitData">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px">
          <div class="stat-card" style="padding: 16px">
            <div class="stat-label">市场现价</div>
            <div class="stat-value text-mono" style="font-size: 24px">{{ exitData.current_price?.toFixed(2) }}</div>
          </div>
          <div class="stat-card" style="padding: 16px" :style="{ borderBottom: '2px solid ' + (exitData.change_pct >= 0 ? 'var(--accent-red)' : 'var(--accent-green)') }">
            <div class="stat-label">当前收益率</div>
            <div class="stat-value text-mono" style="font-size: 24px" :class="exitData.change_pct >= 0 ? 'text-up' : 'text-down'">
              {{ exitData.change_pct > 0 ? '+' : '' }}{{ exitData.change_pct }}%
            </div>
          </div>
        </div>
        
        <h4 class="mb-16" style="color: var(--text-secondary); font-size: 13px; font-weight: 700">核心退出条件</h4>
        <div v-for="(cond, key) in exitData.conditions" :key="key" class="exit-condition-item glass" :class="{ triggered: cond.triggered }">
          <div style="display: flex; justify-content: space-between; align-items: center">
            <span class="label">{{ cond.label }}</span>
            <span :class="cond.triggered ? 'tag tag-buy' : 'tag tag-neutral'">
              {{ cond.triggered ? '已触发' : '监控中' }}
            </span>
          </div>
          <div class="details">
            <span v-if="cond.price">目标价: <span class="text-mono">{{ cond.price }}</span></span>
            <span v-if="cond.holding_days">已持仓: <span class="text-mono">{{ cond.holding_days }} / {{ cond.max_days }}</span> 天</span>
          </div>
        </div>
      </div>
    </el-dialog>

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
import { paperTrading } from '../api'
import { ElMessage } from 'element-plus'
import { Plus, Refresh, Loading, TrendCharts, Aim, Sell } from '@element-plus/icons-vue'

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
const buyForm = ref({ code: '', name: '', buy_date: '', buy_price: 0, quantity: 1 })

// 卖出条件
const exitVisible = ref(false)
const exitLoading = ref(false)
const exitData = ref(null)

// 卖出弹窗
const sellVisible = ref(false)
const sellForm = ref({ position_id: 0, sell_date: '', sell_price: 0, sell_reason: '手动卖出' })

onMounted(() => { loadPositions(); loadHistory() })

async function loadPositions() {
  loading.value = true
  try {
    const { data } = await paperTrading.getPositions('active')
    activePositions.value = data.positions
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
  if (!buyForm.value.code || !buyForm.value.buy_date || !buyForm.value.buy_price) {
    ElMessage.warning('请填写所有必填项')
    return
  }
  buying.value = true
  try {
    await paperTrading.buy(buyForm.value)
    ElMessage.success('成功部署监控')
    buyForm.value = { code: '', name: '', buy_date: '', buy_price: 0, quantity: 1 }
    loadPositions()
  } catch (e) {
    ElMessage.error('记录买入失败')
  } finally {
    buying.value = false
  }
}

async function checkExit(row) {
  exitVisible.value = true
  exitLoading.value = true
  exitData.value = null
  try {
    let params = {}
    const saved = localStorage.getItem('quant_frontend_config')
    if (saved) {
      try {
        const cfg = JSON.parse(saved)
        params = {
          atr_period: cfg.atrPeriod,
          atr_stop_multiplier: cfg.atrStopMultiplier,
          atr_target_multiplier: cfg.atrTargetMultiplier,
          time_stop_days: cfg.timeStopDays,
          time_stop_min_loss_pct: cfg.timeStopMinLossPct
        }
      } catch (e) {}
    }
    const { data } = await paperTrading.checkExit(row.code, row.buy_price, row.buy_date, params)
    exitData.value = data
  } catch (e) {
    ElMessage.error('分析失败')
  } finally {
    exitLoading.value = false
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
