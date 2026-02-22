<template>
  <div class="data-center page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">数据中心</h1>
      <p class="page-subtitle">市场数据持久化 · 同步控制 · 健康监控</p>
    </div>

    <!-- 统计卡片 -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">已索引标的</div>
        <div class="stat-value text-mono">{{ formatNum(status.total_stocks) }}</div>
        <div class="stat-sub">本地引擎跟踪数量</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">最新数据快照</div>
        <div class="stat-value text-mono" style="font-size: 24px; color: var(--accent-green)">{{ status.latest_date ?? '—' }}</div>
        <div class="stat-sub">市场数据同步日期</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">最早数据快照</div>
        <div class="stat-value text-mono" style="font-size: 24px; color: var(--accent-blue)">{{ status.earliest_date ?? '—' }}</div>
        <div class="stat-sub">历史数据起始日期</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px">
      <!-- 数据更新控制 -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">数据同步控制</span>
          <div v-if="updateStatus.running" class="tag tag-buy loading-pulse">正在同步</div>
        </div>
        
        <el-form label-position="top">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px">
            <el-form-item label="扫描范围">
              <el-select v-model="updateForm.mode" style="width: 100%">
                <el-option label="全市场" value="all" />
                <el-option label="指定标的" value="multiple" />
              </el-select>
            </el-form-item>
            <el-form-item label="首选数据源">
              <el-select v-model="updateForm.source" style="width: 100%">
                <el-option label="YFinance (全球)" value="yfinance" />
                <el-option label="AkShare (A股)" value="akshare" />
              </el-select>
            </el-form-item>
          </div>
          
          <el-form-item v-if="updateForm.mode !== 'all'" label="标的代码 (英文逗号分隔)">
            <el-input v-model="symbolInput" placeholder="例: 600000, 000001" />
          </el-form-item>

          <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px">
            <el-checkbox v-model="updateForm.incremental">仅增量更新</el-checkbox>
            <el-button type="primary" size="large" @click="triggerUpdate" :loading="updateStatus.running" :disabled="updateStatus.running">
              <el-icon style="margin-right: 8px"><Refresh /></el-icon> 启动同步
            </el-button>
          </div>
        </el-form>

        <div v-if="updateStatus.progress" class="mt-24 glass" style="padding: 16px; margin-top: 24px">
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px">
            <span style="font-size: 12px; font-weight: 700">当前进度</span>
            <span class="text-mono" style="font-size: 12px">{{ updateStatus.progress }}</span>
          </div>
          <div v-if="updateStatus.error" class="text-up" style="font-size: 11px; margin-top: 8px">检测到错误: {{ updateStatus.error }}</div>
        </div>
      </div>

      <!-- 数据缺失警告 -->
      <div v-if="status.missing_data_info?.length" class="card">
        <div class="card-header">
          <span class="card-title text-up" style="display: flex; align-items: center; gap: 8px">
            <el-icon><Warning /></el-icon> 数据缺失告警
          </span>
        </div>
        <div class="missing-alerts">
          <div v-for="item in status.missing_data_info" :key="item.code" class="missing-item glass">
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span class="text-mono" style="font-weight: 700; color: var(--accent-blue)">{{ item.code }}</span>
              <span class="tag tag-neutral">过期 {{ item.days_ago }} 天</span>
            </div>
            <div style="margin-top: 8px; font-size: 12px; color: var(--text-muted)">
              最后快照: <span class="text-mono">{{ item.last_date }}</span>
            </div>
          </div>
        </div>
      </div>
      

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { dataCenter } from '../api'
import { ElMessage } from 'element-plus'
import { Refresh, Warning, Loading } from '@element-plus/icons-vue'

const loading = ref(false)
const status = ref({})
const updateStatus = ref({ running: false, progress: '', error: null })
const updateForm = ref({ mode: 'all', source: 'yfinance', incremental: true })
const symbolInput = ref('')
let pollTimer = null

const formatNum = (n) => n != null ? n.toLocaleString() : '—'

onMounted(() => { loadStatus(); pollUpdate() })
onUnmounted(() => { if (pollTimer) clearInterval(pollTimer) })

async function loadStatus() {
  loading.value = true
  try {
    const { data } = await dataCenter.getStatus()
    status.value = data
  } catch (e) {
    ElMessage.error('加载数据库状态失败')
  } finally {
    loading.value = false
  }
}

async function triggerUpdate() {
  const params = { ...updateForm.value }
  if (params.mode !== 'all') {
    params.symbols = symbolInput.value.split(',').map(s => s.trim()).filter(Boolean)
    if (!params.symbols.length) {
      ElMessage.warning('请输入标的代码')
      return
    }
  }
  try {
    await dataCenter.triggerUpdate(params)
    ElMessage.success('同步任务已启动')
    pollUpdate()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail ?? '启动同步失败')
  }
}

function pollUpdate() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const { data } = await dataCenter.getUpdateStatus()
      updateStatus.value = data
      if (!data.running) {
        clearInterval(pollTimer)
        loadStatus()
      }
    } catch (e) { /* ignore */ }
  }, 600000)
}
</script>

<style scoped>
.missing-alerts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  max-height: 280px;
  overflow-y: auto;
  padding: 4px;
}
.missing-item {
  padding: 16px;
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-red);
  background: rgba(255, 51, 102, 0.05);
}
</style>
