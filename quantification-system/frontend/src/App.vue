<template>
  <el-config-provider :locale="locale">
    <div class="app-layout">
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
            <el-icon class="nav-icon"><component :is="iconMap[route.meta.icon]" /></el-icon>
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
              <el-icon class="nav-icon"><component :is="iconMap[route.meta.icon]" /></el-icon>
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

      <AuthModal v-model="showAuth" @login-success="onLoginSuccess" />

      <main class="main-content">
        <div v-if="routeLoading" class="route-loading-overlay">
          <el-icon class="loading-pulse" :size="32" color="var(--accent-blue)"><Loading /></el-icon>
        </div>
        <router-view v-slot="{ Component }">
          <transition name="page-fade" mode="out-in">
            <keep-alive :max="3">
              <component :is="Component" />
            </keep-alive>
          </transition>
        </router-view>
      </main>
    </div>
  </el-config-provider>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import { ElMessage } from 'element-plus'
import { User, Avatar, SwitchButton, Menu, TrendCharts, Search, Briefcase, Histogram, Setting, Odometer, Loading } from '@element-plus/icons-vue'
import AuthModal from '@/components/AuthModal.vue'
import { authApi } from '@/api'

const locale = zhCn
const router = useRouter()
const navRoutes = router.getRoutes().filter(r => r.meta?.title)

const iconMap = {
  TrendCharts, Search, Briefcase, Histogram, Setting, Odometer
}

const mobileMenuVisible = ref(false)
const showAuth = ref(false)
const isLoggedIn = ref(false)
const username = ref('访客')
const routeLoading = ref(false)

router.beforeEach(() => { routeLoading.value = true })
router.afterEach(() => { routeLoading.value = false })
router.onError(() => { routeLoading.value = false })

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
    router.go(0)
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
.route-loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
  z-index: 9999;
  backdrop-filter: blur(4px);
}

.page-fade-enter-active,
.page-fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.page-fade-enter-from {
  opacity: 0;
  transform: translateY(6px);
}
.page-fade-leave-to {
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
