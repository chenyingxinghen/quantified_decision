<template>
  <div class="config-center page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">配置中心</h1>
      <p class="page-subtitle">本地独立设置 · 实盘验证与分析参数</p>
    </div>

    <div class="card" style="max-width: 800px; margin: 0 auto">
      <div class="card-header" style="justify-content: space-between; margin-bottom: 24px">
        <span class="card-title">实盘策略退出规则</span>
        <el-button type="primary" @click="saveConfig" style="height: 36px">
          <el-icon style="margin-right: 6px"><Check /></el-icon> 保存设置
        </el-button>
      </div>

      <el-form label-position="top" :model="localConfig" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px">
        <div class="config-item glass">
          <label>ATR 周期</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">真实波动幅度（ATR）的滚动计算窗口</div>
          <el-input-number v-model="localConfig.atrPeriod" :min="5" :max="60" :step="1" style="width: 100%" />
        </div>
        
        <div class="config-item glass">
          <label>ATR 止损乘数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">动态止损线的计算乘数</div>
          <el-input-number v-model="localConfig.atrStopMultiplier" :min="0.5" :max="10.0" :step="0.1" :precision="1" style="width: 100%" />
        </div>
        
        <div class="config-item glass">
          <label>ATR 止盈乘数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">止盈目标的计算乘数</div>
          <el-input-number v-model="localConfig.atrTargetMultiplier" :min="1.0" :max="15.0" :step="0.1" :precision="1" style="width: 100%" />
        </div>

        <div class="config-item glass">
          <label>最大持有天数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">触发时间止损的最大持仓周期</div>
          <el-input-number v-model="localConfig.timeStopDays" :min="1" :max="100" :step="1" style="width: 100%" />
        </div>

        <div class="config-item glass" style="grid-column: span 2">
          <label>时间止损容忍值 (%)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">达到最大持有天数后，若收益低于此阈值则强平</div>
          <el-input-number v-model="localConfig.timeStopMinLossPct" :min="-1.0" :max="0.0" :step="0.01" :precision="2" style="width: 100%" />
        </div>
      </el-form>
      
      <div style="margin-top: 24px; padding: 16px; background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; font-size: 13px; color: var(--text-secondary)">
        <el-icon color="var(--accent-blue)" style="margin-right: 8px; vertical-align: middle;"><InfoFilled /></el-icon>
        此处参数将安全地保存在您的浏览器本地，用于执行实盘信号验证，完全独立于后端模型训练系统。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Check, InfoFilled } from '@element-plus/icons-vue'
import { authApi } from '@/api'

const defaultConfig = {
   atrPeriod: 14,
   atrStopMultiplier: 2.0,
   atrTargetMultiplier: 3.0,
   timeStopDays: 20,
   timeStopMinLossPct: -0.05
}

const localConfig = ref(JSON.parse(JSON.stringify(defaultConfig)))

onMounted(async () => {
    let remoteConfigLoaded = false
    const token = localStorage.getItem('quant_user_token')
    
    // 尝试拉取远端配置
    if (token) {
        try {
            const res = await authApi.getConfig()
            if (res.data && res.data.config_json) {
                const configObj = JSON.parse(res.data.config_json)
                localConfig.value = { ...defaultConfig, ...configObj }
                remoteConfigLoaded = true
            }
        } catch (e) {
            console.error('Failed to load remote config', e)
        }
    }
    
    // 未获取远端配置则使用本地缓存
    if (!remoteConfigLoaded) {
        const saved = localStorage.getItem('quant_frontend_config')
        if (saved) {
            try {
                localConfig.value = { ...defaultConfig, ...JSON.parse(saved) }
            } catch (e) {
                console.error('Failed to parse local config', e)
            }
        }
    }
})

async function saveConfig() {
    const configStr = JSON.stringify(localConfig.value)
    const token = localStorage.getItem('quant_user_token')
    
    let savedToRemote = false
    if (token) {
        try {
            await authApi.saveConfig(configStr)
            savedToRemote = true
        } catch (e) {
            console.error('Failed to save to remote DB', e)
        }
    }
    
    // 如果没有登录，或者远端保存失败，保存到本地
    if (!savedToRemote) {
        localStorage.setItem('quant_frontend_config', configStr)
        ElMessage.success('前端配置已更新并保存在本地')
    } else {
        localStorage.setItem('quant_frontend_config', configStr) // 为了后续兜底
        ElMessage.success('前端配置已同步至账号云端')
    }
}
</script>

<style scoped>
.config-item {
  display: flex;
  flex-direction: column;
  padding: 16px;
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.05);
}
.config-item label {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
}
</style>
