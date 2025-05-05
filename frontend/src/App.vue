<template>
  
  <div class="layout-wrapper">

    <!-- Header内容 -->
    <el-page-header>
      <!-- 标题栏 -->
      <template #content>
        <div class="page-header-title">
          <el-avatar
            class="mr-3"
            :size="32"
            :src="LakeCompassIcon"
          />
          <span class="text-large font-600 mr-3"> &nbsp; Lake Compass </span>
        </div>
      </template>
      <!-- step步进条 -->
      <div class="step-header">
        <el-steps :active="active" finish-status="success" align-center>
          <el-step title="Query Table Information" />
          <el-step title="Candidate Tables" />
          <el-step title="Model Analysis" />
        </el-steps>
      </div>
    </el-page-header>

    <!-- 主页面内容 -->
    <main class="step-main">
      <div class="step-content">
        <Step1 v-if="active === 0" />
        <Step2 v-else-if="active === 1" />
        <Step3 v-else-if="active === 2" />
      </div>
    </main>

    <!-- 底部翻页按钮 -->
    <footer class="step-footer">
      <el-button :icon="ArrowLeft" :disabled="active === 0" @click="prev">
        Previous
      </el-button>
      <el-button type="primary" :disabled="active === 2" @click="next">
        Next<el-icon class="el-icon--right"><ArrowRight /></el-icon>
      </el-button>
    </footer>
  </div>

</template>

<script setup>
import { ref, provide } from 'vue'
import {
  ArrowLeft,
  ArrowRight,
} from '@element-plus/icons-vue'
import Step1 from './components/Step1.vue'
import Step2 from './components/Step2.vue'
import Step3 from './components/Step3.vue'

import LakeCompassIcon from './components/icons/LakeCompass.png'

const active = ref(0)

const next = () => {
  if (active.value < 2) active.value++
}
const prev = () => {
  if (active.value > 0) active.value--
}

// 解决step2和3的selectHeader问题
const globalState = ref({
  filename: '',
  selectedHeader: '',
  model: '',
  task: ''
})

// 状态修改方法
const setFileName = (value) => {
  globalState.value.fileName = value
}

const setSelectedHeader = (value) => {
  globalState.value.selectedHeader = value
}

const setModel = (value) => {
  globalState.value.model = value
}

const setTask = (value) => {
  globalState.value.task = value
}

// 暴露状态给所有子组件
provide('globalState', globalState)
provide('setFileName', setFileName)  // ✅ 提供方法
provide('setSelectedHeader', setSelectedHeader)
provide('setModel', setModel)
provide('setTask', setTask)
</script>

<style scoped>
/* 整体页面 */
.layout-wrapper {
  width: 96vw;
  height: 90vh;
  display: grid;
  grid-template-rows: auto 1fr auto;
  position: relative;
  /* background-color: #f2f2f2; */
  box-sizing: border-box;
}

/* 标题栏 */
.page-header-title {
  display: flex;
  align-items: center; /* 垂直居中对齐 avatar 和文字 */
}

/* 步进条区域 */
.step-header {
  width: 100%;
  background: #fff;
  border-bottom: 1px solid #ddd;
  display: flex;
  align-items: center;        /* 竖直居中 */
  justify-content: center;    /* 水平居中 */
  padding: 20px;              /* 增加高度 + 内边距 */
  box-sizing: border-box;     /* 确保 padding 不撑破 */
}

/* 中间主内容区域 */
.step-main {
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

/* 内容容器自适应居中显示，支持 16:9 缩放 */
.step-content {
  width: 100%;
  height: 100%;
  max-width: 90vw;
  max-height: 90vh;
  aspect-ratio: 16 / 9;
  padding: 20px;
  overflow: auto;
  background-color: #fafafa;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
}

/* 底部按钮浮动固定位置 */
.step-footer {
  width: 100%;
  padding: 10px 5%;
  box-sizing: border-box;
  display: flex;
  justify-content: space-between;
  pointer-events: none; /* 避免遮挡内容 */
}


/* 调整 el-step 宽度为 1/6，防止换行 */
.el-steps {
  display: flex;
  flex-wrap: nowrap;
  /* overflow-x: auto; 可选：若屏幕太小可以横向滚动 */
  flex: 0 0 100%;
  white-space: nowrap;
  text-align: center;
}

.step-footer .el-button {
  pointer-events: all;
}
</style>
