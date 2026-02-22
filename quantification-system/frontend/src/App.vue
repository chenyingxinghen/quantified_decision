<template>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-logo">
        <el-icon :size="24" color="var(--accent-blue)"><TrendCharts /></el-icon>
        <h1>量化决策系统</h1>
      </div>
      <nav class="sidebar-nav">
        <router-link
          v-for="route in navRoutes"
          :key="route.path"
          :to="route.path"
          class="nav-item"
          active-class="active"
        >
          <el-icon class="nav-icon"><component :is="route.meta.icon" /></el-icon>
          <span>{{ route.meta.title }}</span>
        </router-link>
      </nav>
      <div class="sidebar-footer" style="display: flex; flex-direction: column; gap: 8px;">
        <div v-if="!isLoggedIn" class="user-block login-btn" @click="showAuth = true">
          <el-icon><User /></el-icon>
          <span>未登录</span>
        </div>
        <div v-else class="user-block logged-in" @click="handleLogout">
          <el-icon><Avatar /></el-icon>
          <span>{{ username }}</span>
          <el-tooltip content="退出系统" placement="right">
            <el-icon class="logout-icon"><SwitchButton /></el-icon>
          </el-tooltip>
        </div>
        <div class="version">v2.0.0 · AI 深度优化</div>
      </div>
    </aside>

    <!-- 登录注册弹窗 -->
    <AuthModal v-model="showAuth" @login-success="onLoginSuccess" />

    <!-- 主内容区 -->
    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="page-fade" mode="out-in">
          <keep-alive>
            <component :is="Component" />
          </keep-alive>
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { authApi } from '@/api'
import { User, Avatar, SwitchButton } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import AuthModal from '@/components/AuthModal.vue'

const router = useRouter()
const navRoutes = router.getRoutes().filter(r => r.meta?.title)

const showAuth = ref(false)
const isLoggedIn = ref(false)
const username = ref('访客')

onMounted(async () => {
    const token = localStorage.getItem('quant_user_token')
    if (token) {
        try {
            const res = await authApi.getInfo()
            if (res.data.is_logged_in) {
                isLoggedIn.value = true
                username.value = res.data.username
                localStorage.setItem('quant_username', username.value)
            } else {
                handleLogout(false)
            }
        } catch {
            handleLogout(false)
        }
    }
})

function onLoginSuccess(name) {
    isLoggedIn.value = true
    username.value = name
    router.go(0) // Refresh the page to load configurations from backend if needed
}

async function handleLogout(showToast = true) {
    if (isLoggedIn.value && showToast) {
        try {
            await authApi.logout()
            ElMessage.success('已退出登录')
        } catch {}
    }
    localStorage.removeItem('quant_user_token')
    localStorage.removeItem('quant_username')
    isLoggedIn.value = false
    username.value = '访客'
    if (showToast) {
        router.go(0)
    }
}
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.fade-enter-from {
  opacity: 0;
  transform: translateY(6px);
}
.fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.user-block {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s ease;
  background: var(--bg-hover);
}
.user-block:hover {
  background: rgba(0, 242, 254, 0.1);
  color: var(--accent-blue);
}
.logout-icon {
  margin-left: auto;
  opacity: 0.6;
}
.logout-icon:hover {
  opacity: 1;
  color: var(--accent-red);
}
</style>
