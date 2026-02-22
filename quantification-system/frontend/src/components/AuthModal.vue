<template>
  <el-dialog
    v-model="visible"
    :title="isLogin ? '系统登录' : '新用户注册'"
    width="400px"
    center
    :close-on-click-modal="false"
    class="auth-dialog"
  >
    <el-tabs v-model="activeTab" @tab-change="handleTabChange" class="auth-tabs">
      <el-tab-pane label="登录" name="login">
        <el-form :model="loginForm" label-position="top" @submit.prevent="handleLogin">
          <el-form-item label="用户名">
            <el-input v-model="loginForm.username" placeholder="请输入用户名" />
          </el-form-item>
          <el-form-item label="密码">
            <el-input v-model="loginForm.password" type="password" show-password @keyup.enter="handleLogin" placeholder="请输入密码" />
          </el-form-item>
          <el-button type="primary" native-type="submit" style="width: 100%; margin-top: 16px" :loading="loading">
            登录系统
          </el-button>
        </el-form>
      </el-tab-pane>
      
      <el-tab-pane label="注册" name="register">
        <el-form :model="registerForm" label-position="top" @submit.prevent="handleRegister">
          <el-form-item label="用户名">
            <el-input v-model="registerForm.username" placeholder="请输入用户名" />
          </el-form-item>
          <el-form-item label="密码">
            <el-input v-model="registerForm.password" type="password" show-password placeholder="请输入密码" />
          </el-form-item>
          <el-form-item label="用户来源">
            <el-select v-model="registerForm.source" placeholder="请选择您是如何知道我们的" style="width: 100%">
              <el-option label="朋友推荐" value="friend" />
              <el-option label="搜索引擎" value="search" />
              <el-option label="社交媒体" value="social" />
              <el-option label="技术社区" value="community" />
              <el-option label="其他来源" value="other" />
            </el-select>
          </el-form-item>
          <el-button type="success" native-type="submit" style="width: 100%; margin-top: 16px" :loading="loading">
            创建账号
          </el-button>
        </el-form>
      </el-tab-pane>
    </el-tabs>
  </el-dialog>
</template>

<script setup>
import { ref, watch, computed } from 'vue'
import { authApi } from '@/api'
import { ElMessage } from 'element-plus'

const props = defineProps({
  modelValue: Boolean
})
const emit = defineEmits(['update:modelValue', 'login-success'])

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

const activeTab = ref('login')
const isLogin = computed(() => activeTab.value === 'login')
const loading = ref(false)

const loginForm = ref({ username: '', password: '' })
const registerForm = ref({ username: '', password: '', source: '' })

function handleTabChange() {
  loginForm.value = { username: '', password: '' }
  registerForm.value = { username: '', password: '', source: '' }
}

async function handleLogin() {
  if (!loginForm.value.username || !loginForm.value.password) {
    ElMessage.warning('请输入完整登录信息')
    return
  }
  try {
    loading.value = true
    const res = await authApi.login(loginForm.value)
    if (res.data.token) {
      localStorage.setItem('quant_user_token', res.data.token)
      localStorage.setItem('quant_username', res.data.username)
      ElMessage.success('登录成功')
      visible.value = false
      emit('login-success', res.data.username)
    }
  } catch (err) {
    ElMessage.error(err.response?.data?.detail || '登录失败')
  } finally {
    loading.value = false
  }
}

async function handleRegister() {
  if (!registerForm.value.username || !registerForm.value.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  try {
    loading.value = true
    await authApi.register(registerForm.value)
    ElMessage.success('注册成功，请重新登录')
    activeTab.value = 'login'
    loginForm.value.username = registerForm.value.username
  } catch (err) {
    ElMessage.error(err.response?.data?.detail || '注册失败')
  } finally {
    loading.value = false
  }
}
</script>

<style>
/* 继承主页面色调 */
.auth-dialog .el-dialog {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  box-shadow: 0 16px 32px rgba(0, 0, 0, 0.5);
}
.auth-dialog .el-dialog__title {
  color: var(--text-primary);
  font-weight: 600;
}
.auth-tabs .el-tabs__item {
  color: var(--text-secondary);
}
.auth-tabs .el-tabs__item.is-active {
  color: var(--accent-blue);
  font-weight: 600;
}
.auth-tabs .el-tabs__active-bar {
  background-color: var(--accent-blue);
}
</style>
