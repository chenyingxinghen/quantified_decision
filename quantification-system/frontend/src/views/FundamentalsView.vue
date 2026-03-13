<template>
  <div class="fundamentals page-fade-enter">
    <div class="page-header">
      <div style="display: flex; align-items: center; gap: 20px">
        <el-button circle @click="router.back()" class="glass-btn">
          <el-icon><ArrowLeft /></el-icon>
        </el-button>
        <div>
          <h1 class="page-title">基本面深度分析</h1>
          <p class="page-subtitle">财务质量评估 · 成长因子追踪 · 行业专项指标</p>
        </div>
      </div>
      
      <div class="header-search glass">
        <el-select
          v-model="stockCode"
          filterable
          remote
          reserve-keyword
          placeholder="搜索代码/名称"
          :remote-method="queryStock"
          :loading="searchLoading"
          style="width: 280px"
          @change="loadData"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
          <el-option
            v-for="item in searchResults"
            :key="item.code"
            :label="item.code + ' ' + item.name"
            :value="item.code"
          />
        </el-select>
      </div>
    </div>

    <div v-if="loading" class="flex-center" style="height: 60vh">
      <el-icon class="loading-pulse" :size="48" color="var(--accent-blue)"><Loading /></el-icon>
    </div>

    <div v-else-if="!stockData.info" class="flex-center" style="height: 60vh; flex-direction: column">
      <div class="glass-icon-circle mb-24">
        <el-icon :size="40" color="var(--text-muted)"><InfoFilled /></el-icon>
      </div>
      <p style="color: var(--text-muted)">请输入股票代码开始分析</p>
    </div>

    <div v-else class="content-container">
      <!-- 顶部概览 -->
      <div class="summary-card glass mb-24">
        <div class="card-body">
          <div class="main-info">
            <div class="brand">
              <h2 class="stock-name">{{ stockData.info.name }}</h2>
              <div class="stock-meta">
                <span class="code text-mono">{{ stockData.code }}</span>
                <span class="tag">{{ stockData.info.sector }}</span>
                <span class="tag">{{ stockData.info.industry }}</span>
              </div>
            </div>
            
            <div class="live-price">
              <div class="price-val text-mono">{{ stockData.valuation?.close?.toFixed(2) }}</div>
              <div class="price-date text-muted">最新收盘 ({{ stockData.valuation?.date }})</div>
            </div>
          </div>

          <div class="mini-stats">
            <div class="mini-stat-item">
              <div class="label">PE (动态)</div>
              <div class="value text-mono">{{ stockData.valuation?.pe ?? '—' }}</div>
            </div>
            <div class="mini-stat-item">
              <div class="label">PB (动态)</div>
              <div class="value text-mono">{{ stockData.valuation?.pb ?? '—' }}</div>
            </div>
            <div class="mini-stat-item">
              <div class="label">机构类型</div>
              <div class="value text-mono">{{ formatOrgType(stockData.finance?.current?.ORG_TYPE) }}</div>
            </div>
            <div class="mini-stat-item">
              <div class="label">报表日期</div>
              <div class="value text-mono">{{ stockData.finance?.report_date }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="analysis-grid">
        <!-- 盈利分析 -->
        <div class="card glass section-card">
          <div class="card-header">
            <el-icon color="var(--accent-red)"><Histogram /></el-icon>
            <span class="card-title">盈利质量</span>
          </div>
          <div class="card-body">
            <div class="indicator-grid">
              <div class="indicator-item">
                <div class="label">ROE (加权)</div>
                <div class="value text-up">{{ stockData.finance?.current?.ROEJQ?.toFixed(2) }}%</div>
                <div class="trend" :class="getTrendClass(stockData.finance?.current?.ROEJQ, stockData.finance?.previous?.ROEJQ)">
                  {{ formatDiff(stockData.finance?.current?.ROEJQ, stockData.finance?.previous?.ROEJQ) }}
                </div>
              </div>
              <div class="indicator-item">
                <div class="label">ROE (扣非)</div>
                <div class="value text-up">{{ stockData.finance?.current?.ROEKCJQ?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">销售净利率</div>
                <div class="value text-mono">{{ stockData.finance?.current?.XSJLL?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">总资产收益率</div>
                <div class="value text-mono">{{ stockData.finance?.current?.ZZCJLL?.toFixed(2) }}%</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 成长动力 -->
        <div class="card glass section-card">
          <div class="card-header">
            <el-icon color="var(--accent-cyan)"><TrendCharts /></el-icon>
            <span class="card-title">成长动力</span>
          </div>
          <div class="card-body">
            <div class="indicator-grid">
              <div class="indicator-item">
                <div class="label">营收同比</div>
                <div class="value" :class="getValClass(stockData.finance?.current?.TOTALOPERATEREVETZ)">
                  {{ stockData.finance?.current?.TOTALOPERATEREVETZ?.toFixed(2) }}%
                </div>
                <div class="sub-label">预测: {{ stockData.finance?.current?.DJD_TOI_YOY?.toFixed(1) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">净利润同比</div>
                <div class="value" :class="getValClass(stockData.finance?.current?.PARENTNETPROFITTZ)">
                  {{ stockData.finance?.current?.PARENTNETPROFITTZ?.toFixed(2) }}%
                </div>
                <div class="sub-label">预测: {{ stockData.finance?.current?.DJD_DPNP_YOY?.toFixed(1) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">扣非净利同比</div>
                <div class="value" :class="getValClass(stockData.finance?.current?.KCFJCXSYJLRTZ)">
                  {{ stockData.finance?.current?.KCFJCXSYJLRTZ?.toFixed(2) }}%
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 现金流量 -->
        <div class="card glass section-card">
          <div class="card-header">
            <el-icon color="var(--accent-blue)"><Wallet /></el-icon>
            <span class="card-title">现金流量</span>
          </div>
          <div class="card-body">
            <div class="indicator-grid">
              <div class="indicator-item">
                <div class="label">每股经营现金流</div>
                <div class="value text-mono">{{ stockData.finance?.current?.MGJYXJJE?.toFixed(3) }}</div>
              </div>
              <div class="indicator-item">
                <div class="label">经营现金流/营收</div>
                <div class="value text-mono">{{ stockData.finance?.current?.JYXJLYYSR?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">每股收益 (EPS)</div>
                <div class="value text-mono">{{ stockData.finance?.current?.EPSJB?.toFixed(3) }}</div>
              </div>
              <div class="indicator-item">
                <div class="label">每股净资产 (BPS)</div>
                <div class="value text-mono">{{ stockData.finance?.current?.BPS?.toFixed(2) }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 资本结构 -->
        <div class="card glass section-card">
          <div class="card-header">
            <el-icon color="var(--accent-purple)"><Connection /></el-icon>
            <span class="card-title">资本结构与风险</span>
          </div>
          <div class="card-body">
            <div class="indicator-grid">
              <div class="indicator-item">
                <div class="label">资产负债率</div>
                <div class="value" :class="stockData.finance?.current?.ZCFZL > 70 ? 'text-down' : 'text-mono'">
                  {{ stockData.finance?.current?.ZCFZL?.toFixed(2) }}%
                </div>
              </div>
              <div class="indicator-item">
                <div class="label">权益乘数</div>
                <div class="value text-mono">{{ stockData.finance?.current?.QYCS?.toFixed(2) }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 行业专项 (银行业) -->
        <div v-if="isBank" class="card glass section-card highlight" style="grid-column: span 2">
          <div class="card-header">
            <el-icon color="var(--accent-orange)"><OfficeBuilding /></el-icon>
            <span class="card-title">银行业专项指标</span>
          </div>
          <div class="card-body">
            <div class="indicator-grid cols-5">
              <div class="indicator-item">
                <div class="label">不良贷款率</div>
                <div class="value text-mono">{{ stockData.finance?.current?.NONPERLOAN?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">拨备覆盖率</div>
                <div class="value text-mono">{{ stockData.finance?.current?.BLDKBBL?.toFixed(1) ?? '—' }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">一级资本充足率</div>
                <div class="value text-mono">{{ stockData.finance?.current?.FIRST_ADEQUACY_RATIO?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">净息差</div>
                <div class="value text-mono">{{ stockData.finance?.current?.NET_INTEREST_MARGIN?.toFixed(2) }}%</div>
              </div>
              <div class="indicator-item">
                <div class="label">利差</div>
                <div class="value text-mono">{{ stockData.finance?.current?.NET_INTEREST_SPREAD?.toFixed(2) ?? '—' }}%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 历史趋势图 -->
      <div class="card glass mt-24">
        <div class="card-header">
          <el-icon color="var(--accent-blue)"><DataLine /></el-icon>
          <span class="card-title">业绩成长趋势 (近 12 期)</span>
        </div>
        <div class="card-body">
          <div ref="chartRef" style="width: 100%; height: 400px"></div>
        </div>
      </div>
      

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { stockSelector } from '@/api'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { 
  ArrowLeft, Search, Loading, InfoFilled, 
  Histogram, TrendCharts, Wallet, Connection, 
  OfficeBuilding, DataLine, Briefcase
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const stockCode = ref('')
const loading = ref(false)
const stockData = ref({})
const searchLoading = ref(false)
const searchResults = ref([])
const chartRef = ref(null)
let chartInstance = null

const isBank = computed(() => {
  const type = stockData.value.finance?.current?.ORG_TYPE
  return type === 1 || stockData.value.info?.industry?.includes('银行')
})

onMounted(() => {
  if (route.query.code) {
    stockCode.value = route.query.code
    loadData()
  }
})

onUnmounted(() => {
  if (chartInstance) chartInstance.dispose()
})

async function queryStock(query) {
  if (!query) { searchResults.value = []; return }
  searchLoading.value = true
  try {
    const { data } = await stockSelector.search(query)
    searchResults.value = data.items
  } catch (e) {
    console.error(e)
  } finally {
    searchLoading.value = false
  }
}

async function loadData() {
  if (!stockCode.value) return
  loading.value = true
  try {
    const { data } = await stockSelector.getFundamental(stockCode.value)
    stockData.value = data
    
    // 更新 URL，方便刷新
    router.replace({ path: route.path, query: { code: stockCode.value } })
  } catch (e) {
    ElMessage.error('加载基本面数据失败')
  } finally {
    loading.value = false
    // 确保在 loading 状态改变并在 DOM 更新之后渲染图表
    await nextTick()
    renderChart()
  }
}

function renderChart() {
  if (!chartRef.value || !stockData.value.finance?.history) return
  
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(chartRef.value, 'dark')
  
  const history = [...stockData.value.finance.history].reverse()
  const dates = history.map(h => h.REPORT_DATE)
  const revenue = history.map(h => h.TOTALOPERATEREVETZ)
  const profit = history.map(h => h.PARENTNETPROFITTZ)
  const roe = history.map(h => h.ROEJQ)

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20, 24, 33, 0.9)',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      textStyle: { color: '#fff', fontSize: 12 }
    },
    legend: {
      data: ['营收同比', '净利同比', 'ROE'],
      bottom: 0,
      textStyle: { color: '#888' }
    },
    grid: { left: '3%', right: '4%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: '#888', rotate: 30 }
    },
    yAxis: [
      {
        type: 'value',
        name: '同比 (%)',
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        axisLabel: { color: '#888' }
      },
      {
        type: 'value',
        name: 'ROE (%)',
        position: 'right',
        splitLine: { show: false },
        axisLabel: { color: '#888' }
      }
    ],
    series: [
      {
        name: '营收同比',
        type: 'bar',
        data: revenue,
        itemStyle: { color: 'rgba(0, 242, 254, 0.6)' }
      },
      {
        name: '净利同比',
        type: 'bar',
        data: profit,
        itemStyle: { color: 'rgba(255, 51, 102, 0.6)' }
      },
      {
        name: 'ROE',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        data: roe,
        itemStyle: { color: '#FFD700' },
        lineStyle: { width: 3 }
      }
    ]
  }
  
  chartInstance.setOption(option)
  window.addEventListener('resize', () => chartInstance?.resize())
}

function formatOrgType(type) {
  const dict = { 1: '银行', 2: '证券', 3: '保险', 4: '工业/通用', 5: '商业' }
  return dict[type] || (type === 0 ? '通用' : '—')
}

function getTrendClass(curr, prev) {
  if (curr == null || prev == null) return ''
  return curr >= prev ? 'trend-up' : 'trend-down'
}

function formatDiff(curr, prev) {
  if (curr == null || prev == null) return ''
  const diff = curr - prev
  return (diff >= 0 ? '+' : '') + diff.toFixed(2) + '%'
}

function getValClass(val) {
  if (val == null) return 'text-mono'
  return val >= 0 ? 'text-up' : 'text-down'
}

</script>

<style scoped>
.fundamentals {
  padding-bottom: 40px;
}

.header-search {
  padding: 6px 12px;
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
}

.content-container {
  max-width: 1400px;
  margin: 0 auto;
}

.summary-card .card-body {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 32px;
}

.stock-name {
  font-size: 36px;
  font-weight: 800;
  color: var(--accent-blue);
  margin-bottom: 8px;
}

.stock-meta {
  display: flex;
  gap: 12px;
  align-items: center;
}

.stock-meta .code {
  font-size: 18px;
  color: var(--text-muted);
}

.price-val {
  font-size: 44px;
  font-weight: 800;
  line-height: 1;
  text-align: right;
}

.mini-stats {
  display: flex;
  gap: 32px;
  padding-left: 48px;
  border-left: 1px solid var(--border-color);
}

.mini-stat-item .label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
}

.mini-stat-item .value {
  font-size: 18px;
  font-weight: 700;
  color: var(--accent-blue);
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

.section-card {
  height: 100%;
}

.indicator-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.indicator-grid.cols-5 {
  grid-template-columns: repeat(5, 1fr);
}

.indicator-item {
  padding: 16px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.indicator-item .label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.indicator-item .value {
  font-size: 24px;
  font-weight: 800;
  font-family: var(--font-mono);
}

.indicator-item .trend {
  font-size: 11px;
  margin-top: 4px;
  font-weight: 600;
}

.trend-up { color: var(--accent-red); }
.trend-down { color: var(--accent-green); }

.indicator-item .sub-label {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
}

.highlight {
  border: 1px solid rgba(255, 170, 0, 0.3);
  background: linear-gradient(135deg, rgba(255, 170, 0, 0.05) 0%, transparent 100%);
}

.glass-icon-circle {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-color);
}
</style>
