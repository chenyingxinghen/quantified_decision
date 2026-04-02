<template>
  <div class="app-layout">
    <!-- 移动端顶部状态栏 -->
    <header class="mobile-header">
      <div class="mobile-logo-wrap">
        <el-icon :size="20" color="var(--accent-blue)"><TrendCharts /></el-icon>
        <span class="mobile-logo-text">量化决策系统</span>
      </div>
      <div class="mobile-actions">
        <el-button class="mobile-menu-btn" @click="mobileMenuVisible = true">
          <el-icon><Menu /></el-icon>
        </el-button>
      </div>
    </header>

    <!-- 侧边栏 (桌面端) -->
    <aside class="sidebar desktop-only">
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
      <div class="sidebar-footer">
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

    <!-- 移动端抽屉 (Drawer) -->
    <el-drawer
      v-model="mobileMenuVisible"
      direction="ltr"
      size="260px"
      :with-header="false"
      class="mobile-nav-drawer"
    >
      <div class="drawer-header">
        <el-icon :size="20" color="var(--accent-blue)"><TrendCharts /></el-icon>
        <span>量化决策系统</span>
      </div>
      <div class="drawer-content">
        <nav class="sidebar-nav drawer-nav">
          <router-link
            v-for="route in navRoutes"
            :key="route.path"
            :to="route.path"
            class="nav-item"
            active-class="active"
            @click="mobileMenuVisible = false"
          >
            <el-icon class="nav-icon"><component :is="route.meta.icon" /></el-icon>
            <span>{{ route.meta.title }}</span>
          </router-link>
        </nav>
      </div>
      <div class="drawer-footer">
        <div v-if="!isLoggedIn" class="user-block login-btn" @click="showAuth = true; mobileMenuVisible = false">
          <el-icon><User /></el-icon>
          <span>未登录</span>
        </div>
        <div v-else class="user-block logged-in" @click="handleLogout">
          <el-icon><Avatar /></el-icon>
          <span>{{ username }}</span>
          <el-icon class="logout-icon"><SwitchButton /></el-icon>
        </div>
      </div>
    </el-drawer>

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
import { User, Avatar, SwitchButton, Menu, TrendCharts } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import AuthModal from '@/components/AuthModal.vue'

const router = useRouter()
const navRoutes = router.getRoutes().filter(r => r.meta?.title)

const mobileMenuVisible = ref(false)
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

/* 响应式样式补丁 */
.mobile-header {
  display: none;
  height: 56px;
  background: rgba(10, 10, 10, 0.8);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  padding: 0 16px;
  align-items: center;
  justify-content: space-between;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1001;
}

.mobile-logo-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
}

.mobile-logo-text {
  font-size: 15px;
  font-weight: 700;
  color: #fff;
}

.mobile-menu-btn {
  background: transparent !important;
  border: 1px solid var(--border-color) !important;
  color: var(--text-primary) !important;
}

.drawer-header {
  padding: 24px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border-color);
  font-weight: 700;
  font-size: 16px;
}

.drawer-content {
  flex: 1;
  padding: 12px 6px;
}

.drawer-footer {
  padding: 16px;
  border-top: 1px solid var(--border-color);
}

.sidebar-footer {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

@media (max-width: 768px) {
  .desktop-only {
    display: none !important;
  }
  .mobile-header {
    display: flex;
  }
}
</style>
