<template>
  <div class="analysis page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">量化分析引擎</h1>
      <p class="page-subtitle">交互式 K 线图 · 多周期趋势线 · 图表形态识别</p>
    </div>

    <!-- 搜索栏 -->
    <div class="card mb-24 glass-search-bar" style="padding: 16px 24px">
      <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap">
        <div style="display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 700; color: var(--text-secondary)">标的代码</span>
          <el-select
            v-model="stockCode"
            filterable
            remote
            reserve-keyword
            placeholder="代码/名称"
            :remote-method="queryStock"
            :loading="searchLoading"
            style="width: 180px"
            @change="loadChart"
          >
            <el-option
              v-for="item in searchResults"
              :key="item.code"
              :label="item.code + ' ' + item.name"
              :value="item.code"
            />
          </el-select>
          <el-tooltip content="基本面分析" placement="top">
            <el-button 
              v-if="stockCode" 
              circle 
              size="small" 
              type="warning" 
              @click="$router.push({ path: '/fundamental', query: { code: stockCode } })"
            >
              <el-icon><Histogram /></el-icon>
            </el-button>
          </el-tooltip>
        </div>
        <div style="display: flex; align-items: center; gap: 12px">
          <span style="font-size: 13px; font-weight: 700; color: var(--text-secondary)">历史范围</span>
          <el-select v-model="days" style="width: 110px">
            <el-option :value="60" label="60 天" />
            <el-option :value="120" label="120 天" />
            <el-option :value="250" label="250 天" />
            <el-option :value="500" label="500 天" />
          </el-select>
        </div>
        
        <div class="v-divider"></div>

        <div style="display: flex; align-items: center; gap: 8px">
          <span style="font-size: 11px; font-weight: 700; color: var(--text-muted)">长期趋势</span>
          <el-input-number v-model="longPeriod" :min="50" :max="500" :step="10" size="small" style="width: 90px" />
        </div>
        <div style="display: flex; align-items: center; gap: 8px">
          <span style="font-size: 11px; font-weight: 700; color: var(--text-muted)">短期趋势</span>
          <el-input-number v-model="shortPeriod" :min="10" :max="100" :step="5" size="small" style="width: 90px" />
        </div>
        
        <div class="v-divider"></div>

        <el-checkbox v-model="showTrendlines" @change="loadChart">显示趋势</el-checkbox>
        <el-checkbox v-model="showPatterns" @change="loadChart">显示信号</el-checkbox>

        <el-button type="primary" @click="loadChart" :loading="loading" style="margin-left: auto; padding: 0 24px">
          <el-icon style="margin-right: 8px"><TrendCharts /></el-icon> 执行分析
        </el-button>
      </div>
    </div>

    <!-- K 线图 -->
    <div class="card" style="padding: 24px; min-height: 550px">
      <div v-if="loading" class="flex-center" style="height: 500px">
        <el-icon class="loading-pulse" :size="48" color="var(--accent-blue)"><Loading /></el-icon>
      </div>
      <div v-else-if="!klineData" class="flex-center" style="height: 500px; flex-direction: column">
        <el-icon :size="64" style="color: var(--border-color); margin-bottom: 20px"><Monitor /></el-icon>
        <p style="color: var(--text-muted); font-size: 14px">输入标的代码以初始化可视化分析</p>
      </div>
      <div ref="chartRef" style="width: 100%; height: 500px" v-show="klineData && !loading"></div>
    </div>

    <!-- 形态识别面板 -->
    <div v-if="patterns" style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px">
      <div class="card">
        <div class="card-header">
          <span class="card-title">信号识别</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px">
          <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px">
              <div style="width: 8px; height: 8px; border-radius: 50%; background: var(--accent-red)"></div>
              <span style="font-size: 13px; font-weight: 700; color: var(--accent-red)">看多形态</span>
            </div>
            <div v-for="p in patterns.bullish_patterns.filter(x => x.date === klineData.dates[klineData.dates.length-1])" :key="p.description" class="pattern-tag bullish">
              <span>{{ p.description }}</span>
              <span class="score">{{ p.score }}</span>
            </div>
            <div v-if="!patterns.bullish_patterns?.filter(x => x.date === klineData.dates[klineData.dates.length-1]).length" class="text-muted" style="font-size: 12px">未检测到显著信号</div>
          </div>
          <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px">
              <div style="width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green)"></div>
              <span style="font-size: 13px; font-weight: 700; color: var(--accent-green)">看空形态</span>
            </div>
            <div v-for="p in patterns.bearish_patterns.filter(x => x.date === klineData.dates[klineData.dates.length-1])" :key="p.description" class="pattern-tag bearish">
              <span>{{ p.description }}</span>
              <span class="score">{{ p.score }}</span>
            </div>
            <div v-if="!patterns.bearish_patterns?.filter(x => x.date === klineData.dates[klineData.dates.length-1]).length" class="text-muted" style="font-size: 12px">未检测到显著信号</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">市场微观结构</span>
          <span v-if="patterns.market_structure?.structure_shift" class="tag tag-buy">结构反转</span>
        </div>
        <div v-if="patterns.market_structure" class="structure-grid">
          <div class="structure-item">
            <span class="label">核心趋势</span>
            <span class="value" :style="{ color: patterns.market_structure.trend?.includes('UP') ? 'var(--accent-red)' : 'var(--accent-green)' }">
              {{ patterns.market_structure.trend?.toUpperCase() ?? 'NEUTRAL' }}
            </span>
          </div>
          <div class="structure-item">
            <span class="label">动量强度</span>
            <span class="value text-mono">{{ patterns.market_structure.strength ?? '0%' }}</span>
          </div>
          <div class="structure-item">
            <span class="label">当前阶段</span>
            <span class="value">{{ patterns.market_structure.pattern?.toUpperCase() ?? 'UNKNOWN' }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { analysis, stockSelector } from '../api'
import { ElMessage } from 'element-plus'
import { TrendCharts, Loading, Monitor, Histogram } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const route = useRoute()
const chartRef = ref(null)
const stockCode = ref('')
const days = ref(250)
const longPeriod = ref(120)
const shortPeriod = ref(20)
const loading = ref(false)
const showTrendlines = ref(true)
const showPatterns = ref(true)

const klineData = ref(null)
const trendlineData = ref(null)
const patterns = ref(null)

const searchLoading = ref(false)
const searchResults = ref([])

let chartInstance = null

onMounted(() => {
  if (route.query.code) {
    stockCode.value = route.query.code
    loadChart()
  }
})

onUnmounted(() => {
  if (chartInstance) chartInstance.dispose()
})

watch(() => route.query.code, (code) => {
  if (code) {
    stockCode.value = code
    loadChart()
  }
})

watch([days, longPeriod, shortPeriod], () => {
  if (stockCode.value) loadChart()
})

async function queryStock(query) {
  if (query) {
    searchLoading.value = true
    try {
      const { data } = await stockSelector.search(query)
      searchResults.value = data.items
    } catch (e) {
      console.error(e)
    } finally {
      searchLoading.value = false
    }
  } else {
    searchResults.value = []
  }
}

async function loadChart() {
  if (!stockCode.value) { ElMessage.warning('请输入标的代码'); return }
  loading.value = true
  try {
    const promises = [analysis.getKline(stockCode.value, days.value)]
    if (showTrendlines.value) {
      promises.push(analysis.getTrendlines(stockCode.value, days.value, longPeriod.value, shortPeriod.value))
    }
    if (showPatterns.value) promises.push(analysis.getPatterns(stockCode.value, days.value))

    const results = await Promise.allSettled(promises)
    if (results[0].status === 'fulfilled') klineData.value = results[0].value.data
    
    let trendIdx = 1
    if (showTrendlines.value && results[trendIdx]?.status === 'fulfilled') {
      trendlineData.value = results[trendIdx].value.data
      trendIdx++
    }
    
    if (showPatterns.value && results[trendIdx]?.status === 'fulfilled') {
      patterns.value = results[trendIdx].value.data
    }

    await nextTick()
    renderChart()
  } catch (e) {
    ElMessage.error('分析失败: ' + (e.response?.data?.detail ?? e.message))
  } finally {
    loading.value = false
  }
}

async function loadPatterns() {
  if (!stockCode.value || !showPatterns.value) { patterns.value = null; return }
  try {
    const { data } = await analysis.getPatterns(stockCode.value, days.value)
    patterns.value = data
  } catch (e) { /* ignore */ }
}

function renderChart() {
  if (!klineData.value || !chartRef.value) return

  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(chartRef.value, 'dark')

  const { dates, values, volumes } = klineData.value
  const upColor = '#ff3366'
  const downColor = '#00f2fe'

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', lineStyle: { color: 'rgba(255,255,255,0.2)' } },
      backgroundColor: 'rgba(20, 24, 33, 0.9)',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      textStyle: { color: '#fff', fontSize: 12 },
      formatter: (params) => {
        let res = `<div style="font-weight:700; margin-bottom: 5px">${params[0].name}</div>`
        params.forEach(p => {
          if (p.seriesName === 'K线') {
            const val = p.value
            res += `O: <span class="text-mono">${val[1]?.toFixed(2)}</span> C: <span class="text-mono">${val[2]?.toFixed(2)}</span><br/>`
            res += `L: <span class="text-mono">${val[3]?.toFixed(2)}</span> H: <span class="text-mono">${val[4]?.toFixed(2)}</span>`
          } else if (p.seriesName === 'VOLUME') {
            res += `<br/>V: <span class="text-mono">${p.value.toLocaleString()}</span>`
          }
        })
        return res
      }
    },
    grid: [
      { left: 50, right: 30, top: 20, height: '65%' },
      { left: 50, right: 30, top: '78%', height: '15%' },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#888', fontSize: 10 }, boundaryGap: true },
      { type: 'category', data: dates, gridIndex: 1, axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { show: false }, boundaryGap: true },
    ],
    yAxis: [
      { scale: true, gridIndex: 0, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }, axisLabel: { color: '#888', fontSize: 10 } },
      { scale: true, gridIndex: 1, splitLine: { show: false }, axisLabel: { show: false } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: 70, end: 100 },
      { type: 'slider', xAxisIndex: [0, 1], start: 70, end: 100, height: 16, bottom: 5, borderColor: 'transparent', fillerColor: 'rgba(255,255,255,0.05)', handleSize: '0', textStyle: { color: 'transparent' } },
    ],
    series: [
      {
        name: 'K线',
        type: 'candlestick',
        data: values,
        itemStyle: { color: upColor, color0: downColor, borderColor: upColor, borderColor0: downColor },
        markPoint: {
          symbol: 'pin',
          symbolSize: 10,
          data: []
        }
      },
      {
        name: '信号',
        type: 'scatter',
        data: [],
        symbolSize: 8,
        label: { show: false }
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes.map((v, i) => ({
          value: v,
          itemStyle: { color: values[i][0] <= values[i][1] ? 'rgba(255,51,102,0.3)' : 'rgba(0,242,254,0.3)' },
        })),
      },
    ],
  }

  // 注入 K 线形态标注
  if (showPatterns.value && patterns.value) {
    const markData = []
    const { bullish_patterns, bearish_patterns } = patterns.value
    
    const cutoffIndex = Math.max(0, dates.length - 90)

    if (bullish_patterns) {
      bullish_patterns.forEach(p => {
        const d = p.date?.split('T')[0] || p.date
        const idx = dates.indexOf(d)
        if (idx >= cutoffIndex) {
          markData.push({
            name: p.description,
            coord: [d, values[idx][2]], // values[idx][2] is low
            value: p.description,
            symbol: 'arrow',
            symbolSize: 15,
            symbolRotate: 0,
            symbolOffset: [0, 10],
            itemStyle: { color: '#ff3366' },
            label: { show: true, position: 'bottom', fontSize: 10, color: '#ff3366', formatter: '{c}' }
          })
        }
      })
    }
    
    if (bearish_patterns) {
      bearish_patterns.forEach(p => {
        const d = p.date?.split('T')[0] || p.date
        const idx = dates.indexOf(d)
        if (idx >= cutoffIndex) {
          markData.push({
            name: p.description,
            coord: [d, values[idx][3]], // values[idx][3] is high
            value: p.description,
            symbol: 'arrow',
            symbolSize: 15,
            symbolRotate: 180,
            symbolOffset: [0, -10],
            itemStyle: { color: '#00f2fe' },
            label: { show: true, position: 'top', fontSize: 10, color: '#00f2fe', formatter: '{c}' }
          })
        }
      })
    }
    option.series[0].markPoint.data = markData
  }

  if (showTrendlines.value && trendlineData.value) {
    const td = trendlineData.value
    const lines = [
      { key: 'uptrend_line', color: 'rgba(255,51,102,0.9)', name: '长期支撑' },
      { key: 'downtrend_line', color: 'rgba(0,242,254,0.9)', name: '长期阻力' },
      { key: 'short_uptrend_line', color: 'rgba(121,82,242,0.9)', name: '短期支撑' },
      { key: 'short_downtrend_line', color: 'rgba(255,170,0,0.9)', name: '短期阻力' },
    ]

    lines.forEach(({ key, color, name }) => {
      const line = td[key]
      if (line && line.valid && line.start_date && line.end_date) {
        let startD = line.start_date
        let endD = line.end_date
        if (typeof startD === 'string') startD = startD.split('T')[0]
        if (typeof endD === 'string') endD = endD.split('T')[0]
        
        option.series[0].markLine = option.series[0].markLine || { data: [], symbol: ['none', 'none'], label: { show: false }, animation: false }
        option.series[0].markLine.data.push([
          { coord: [startD, line.start_price], lineStyle: { color, width: 2, type: 'solid' }, name },
          { coord: [endD, line.end_price] }
        ])
      }
    })
  }

  chartInstance.setOption(option)
  const resizeOb = new ResizeObserver(() => chartInstance?.resize())
  resizeOb.observe(chartRef.value)
}
</script>

<style scoped>
.glass-search-bar {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.v-divider {
  width: 1px;
  height: 24px;
  background: rgba(255, 255, 255, 0.1);
  margin: 0 10px;
}
.pattern-tag {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-radius: 8px;
  margin-bottom: 10px;
  font-size: 13px;
  border-left: 3px solid transparent;
  transition: all 0.3s;
  background: rgba(255, 255, 255, 0.02);
}
.pattern-tag.bullish { border-left-color: var(--accent-red); }
.pattern-tag.bearish { border-left-color: var(--accent-green); }
.pattern-tag:hover { background: rgba(255, 255, 255, 0.05); }
.pattern-tag .score {
  font-weight: 800;
  font-family: var(--font-mono);
  font-size: 11px;
  opacity: 0.8;
}
.structure-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 20px;
}
.structure-item {
  background: rgba(255, 255, 255, 0.02);
  padding: 16px;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.structure-item .label {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-secondary);
  letter-spacing: 0.5px;
}
.structure-item .value {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 700;
}
</style>
