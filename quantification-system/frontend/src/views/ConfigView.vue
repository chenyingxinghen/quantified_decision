<template>
  <div class="config-center page-fade-enter">
    <div class="page-header">
      <h1 class="page-title">系统配置中心</h1>
      <p class="page-subtitle">全局策略参数 · 基本面选股过滤设 · 退出策略参数</p>
    </div>

    <div class="card" style="max-width: 900px; margin: 0 auto 24px auto">
      <div class="card-header" style="justify-content: space-between; margin-bottom: 24px">
        <span class="card-title">基本面选股过滤</span>
        <el-switch
          v-model="config.ENABLE_FUNDAMENTAL_FILTER"
          active-text="启用"
          inactive-text="停用"
          style="--el-switch-on-color: var(--accent-blue)"
        />
      </div>

      <el-form label-position="top" :model="config" :disabled="!config.ENABLE_FUNDAMENTAL_FILTER" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px">
        <div class="config-item glass">
          <label>最小市值 (亿元)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">剔除市值过小的公司</div>
          <el-input-number v-model="config.MIN_MARKET_CAP" :min="0" :max="10000" :step="10" style="width: 100%" />
        </div>
        <div class="config-item glass">
          <label>最大市盈率 (PE)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">剔除估值过低或过高的公司(0或极大值)</div>
          <el-input-number v-model="config.MAX_PE" :min="0" :max="1000" :step="10" style="width: 100%" />
        </div>
        <div class="config-item glass">
          <label>最小股价 (元)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">过滤低价股</div>
          <el-input-number v-model="config.MIN_PRICE" :min="0" :max="500" :step="1" style="width: 100%" />
        </div>
        <div class="config-item glass">
          <label>最大股价 (元)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">过滤高价股</div>
          <el-input-number v-model="config.MAX_PRICE" :min="1" :max="5000" :step="10" style="width: 100%" />
        </div>
        <div class="config-item glass">
          <label>最大资产负债率 (%)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">剔除财务风险较高的公司</div>
          <el-input-number v-model="config.MAX_ZCFZL" :min="0" :max="100" :step="5" style="width: 100%" />
        </div>
        <div class="config-item glass">
          <label>股票市场</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">选择要扫描的市场</div>
          <el-select v-model="config.SELECTOR_MARKETS" multiple collapse-tags placeholder="全选" style="width: 100%">
            <el-option label="沪市主板" value="sh" />
            <el-option label="深市主板" value="sz_main" />
            <el-option label="创业板" value="sz_gem" />
            <el-option label="北交所" value="bj" />
          </el-select>
        </div>
        <div class="config-item glass" style="grid-column: span 2">
          <label>ST 股过滤</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">是否在选股池中包含 ST 股票</div>
          <el-switch v-model="config.INCLUDE_ST" active-text="包含ST股" inactive-text="排除ST股" style="--el-switch-on-color: var(--accent-red)"/>
        </div>
      </el-form>
    </div>

    <!-- 策略退出参数 -->
    <div class="card" style="max-width: 900px; margin: 0 auto">
      <div class="card-header" style="justify-content: space-between; margin-bottom: 24px">
        <span class="card-title">实盘策略追踪与退出规则</span>
        <el-button type="primary" @click="saveConfig" style="height: 36px" :loading="saving">
          <el-icon style="margin-right: 6px"><Check /></el-icon> 保存全部设置
        </el-button>
      </div>

      <div style="margin-bottom: 20px; display: flex; gap: 20px; flex-wrap: wrap;">
        <el-switch v-model="config.ENABLE_STOP_LOSS_EXIT" active-text="启用ATR止损监控" />
        <el-switch v-model="config.ENABLE_TAKE_PROFIT_EXIT" active-text="启用ATR止盈监控" />
        <el-switch v-model="config.ENABLE_SUPPORT_BREAK_EXIT" active-text="启用跌破支撑卖出" />
        <el-switch v-model="config.ENABLE_TIME_STOP_EXIT" active-text="启用时间止损监控" />
      </div>

      <el-form label-position="top" :model="config" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px">
        <div class="config-item glass">
          <label>ATR 周期</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">真实波动幅度计算窗口</div>
          <el-input-number v-model="config.ATR_PERIOD" :min="5" :max="60" :step="1" style="width: 100%" />
        </div>
        
        <div class="config-item glass">
          <label>ATR 止损乘数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">动态止损目标步进 (建议 1.0~3.0)</div>
          <el-input-number v-model="config.ATR_STOP_MULTIPLIER" :min="0.5" :max="10.0" :step="0.1" :precision="1" style="width: 100%" :disabled="!config.ENABLE_STOP_LOSS_EXIT" />
        </div>
        
        <div class="config-item glass">
          <label>ATR 止盈乘数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">预期止盈目标步进 (建议 3.0~6.0)</div>
          <el-input-number v-model="config.ATR_TARGET_MULTIPLIER" :min="1.0" :max="15.0" :step="0.1" :precision="1" style="width: 100%" :disabled="!config.ENABLE_TAKE_PROFIT_EXIT" />
        </div>

        <div class="config-item glass">
          <label>最大持有天数</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">时间耗尽则强平</div>
          <el-input-number v-model="config.TIME_STOP_DAYS" :min="1" :max="100" :step="1" style="width: 100%" :disabled="!config.ENABLE_TIME_STOP_EXIT" />
        </div>

        <div class="config-item glass" style="grid-column: span 2">
          <label>时间止损容忍值 (%)</label>
          <div class="text-muted" style="font-size: 11px; margin-bottom: 8px">持仓到期且收益地域此阈值时强平</div>
          <el-input-number v-model="config.TIME_STOP_MIN_LOSS_PCT" :min="-1.0" :max="0.0" :step="0.01" :precision="2" style="width: 100%" :disabled="!config.ENABLE_TIME_STOP_EXIT" />
        </div>
      </el-form>
      
      <div style="margin-top: 24px; padding: 16px; background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; font-size: 13px; color: var(--text-secondary)">
        <el-icon color="var(--accent-blue)" style="margin-right: 8px; vertical-align: middle;"><InfoFilled /></el-icon>
        配置将保存至您的个人账户（或本地浏览器），对实盘监控和智能选股逻辑生效。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Check, InfoFilled } from '@element-plus/icons-vue'
import { authApi } from '@/api'

const saving = ref(false)

const config = ref({
   ATR_PERIOD: 14,
   ATR_STOP_MULTIPLIER: 2.0,
   ATR_TARGET_MULTIPLIER: 3.0,
   ENABLE_STOP_LOSS_EXIT: true,
   ENABLE_TAKE_PROFIT_EXIT: true,
   ENABLE_SUPPORT_BREAK_EXIT: false,
   ENABLE_TIME_STOP_EXIT: true,
   TIME_STOP_DAYS: 20,
   TIME_STOP_MIN_LOSS_PCT: -0.05,
   
   ENABLE_FUNDAMENTAL_FILTER: false,
   MIN_MARKET_CAP: 80,
    MAX_PE: 80,
    MAX_ZCFZL: 70,
    MIN_PRICE: 1,
    MAX_PRICE: 200,
    INCLUDE_ST: false,
    SELECTOR_MARKETS: ['sh', 'sz_main', 'sz_gem']
})

onMounted(async () => {
    try {
        const res = await authApi.getConfig()
        if (res.data && res.data.config_json) {
            const saved = typeof res.data.config_json === 'string' 
                ? JSON.parse(res.data.config_json) 
                : res.data.config_json
            Object.assign(config.value, saved)
        }
    } catch (e) {
        console.error('Failed to load user config', e)
        // Fallback to default
    }
})

async function saveConfig() {
    saving.value = true
    try {
        await authApi.saveConfig(config.value)
        ElMessage.success('个人偏好设置已保存')
    } catch (e) {
        console.error('Failed to save', e)
        ElMessage.error('保存失败')
    } finally {
        saving.value = false
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
