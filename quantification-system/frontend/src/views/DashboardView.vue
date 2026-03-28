<template>
  <div class="dashboard page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">系统总览</h1>
      <p class="page-subtitle">实时市场数据状态及核心监控指标</p>
    </div>

    <!-- 统计大网格 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">市场覆盖</div>
        <div class="stat-value">{{ formatNum(dbStatus.total_stocks) }} <small style="font-size: 14px; color: var(--text-muted)">只标的</small></div>
        <div class="stat-sub">最新同步: {{ dbStatus.latest_date ?? '—' }}</div>
      </div>
      
      <div class="stat-card" style="border-bottom: 2px solid var(--accent-blue)">
        <div class="stat-label">系统状态</div>
        <div class="stat-value text-mono" style="font-size: 20px; color: var(--accent-green)">运行稳定</div>
        <div class="stat-sub">数据节点正常运转</div>
      </div>

      <div class="stat-card" style="border-bottom: 2px solid var(--accent-purple)">
        <div class="stat-label">AI 选股池</div>
        <div class="stat-value">{{ formatNum(selectionCount) }}</div>
        <div class="stat-sub">{{ selectionFile ? selectionFile.slice(-15) : '暂无近期结果' }}</div>
      </div>

      <div class="stat-card" style="border-bottom: 2px solid var(--accent-green)">
        <div class="stat-label">实盘追踪</div>
        <div class="stat-value">{{ formatNum(activePositions) }}</div>
        <div class="stat-sub">当前活跃监控标的</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px">
      <!-- 快捷操作 -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">快捷指令</span>
        </div>
        <div class="flex-center" style="gap: 20px; flex-wrap: wrap; justify-content: flex-start">
          <div class="action-btn glass" @click="$router.push('/selector')">
            <el-icon><Search /></el-icon>
            <span>启动选股</span>
          </div>
          <div class="action-btn glass" @click="$router.push('/analysis')">
            <el-icon><TrendCharts /></el-icon>
            <span>技术分析</span>
          </div>
          <div class="action-btn glass" @click="$router.push('/paper-trading')">
            <el-icon><Briefcase /></el-icon>
            <span>实盘监控</span>
          </div>

        </div>
      </div>

      <!-- 数据健康 -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">数据监控</span>
        </div>
        <div v-if="dbStatus.missing_data_info?.length" class="text-down" style="font-size: 13px">
          <div v-for="info in dbStatus.missing_data_info" :key="info.code" class="mb-16">
            <span class="text-mono">{{ info.code }}</span> 
            <span class="text-muted" style="margin-left: 8px">自 {{ info.last_date }} 起未更新</span>
          </div>
        </div>
        <div v-else class="text-up" style="display: flex; align-items: center; gap: 8px">
          <el-icon><Check /></el-icon>
          <span>所有核心标的数据已为最新</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Check, Search, TrendCharts, Briefcase, Coin } from '@element-plus/icons-vue'
import { dataCenter, stockSelector, paperTrading } from '../api'

const dbStatus = ref({})
const selectionCount = ref(0)
const selectionFile = ref(null)
const activePositions = ref(0)

const formatNum = (n) => n != null ? n.toLocaleString() : '—'

onMounted(async () => {
  try {
    const cachedDbStatus = localStorage.getItem('quant_db_status')
    if (cachedDbStatus) {
      dbStatus.value = JSON.parse(cachedDbStatus)
    }

    const promises = [
      stockSelector.getLatest(),
      paperTrading.getPositions('active'),
    ]
    promises.push(dataCenter.getStatus())

    const results = await Promise.allSettled(promises)
    const [selRes, posRes, dbRes] = results
    
    if (selRes.status === 'fulfilled') {
      selectionCount.value = selRes.value.data.items?.length ?? 0
      selectionFile.value = selRes.value.data.file
    }
    if (posRes.status === 'fulfilled') {
      activePositions.value = posRes.value.data.positions?.length ?? 0
    }
    if (dbRes && dbRes.status === 'fulfilled') {
      dbStatus.value = dbRes.value.data
      localStorage.setItem('quant_db_status', JSON.stringify(dbRes.value.data))
    }
  } catch (e) {
    console.error(e)
  }
})
</script>

<style scoped>
.action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 24px;
  width: 140px;
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid rgba(255, 255, 255, 0.05);
}
.action-btn:hover {
  background: rgba(var(--accent-blue-rgb), 0.1);
  border-color: var(--accent-blue);
  transform: translateY(-4px);
}
.action-btn i {
  font-size: 24px;
  color: var(--accent-blue);
}
.action-btn span {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
}
</style>
