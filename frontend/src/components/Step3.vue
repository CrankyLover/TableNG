<template>
    <h3>Model Selection</h3>

    <el-container style="height: 35%;">
      <!-- 左侧 Main -->
      <el-main style="padding: 20px;">
        <el-row :gutter="20">
          <!-- 第一个：模型选择 -->
          <el-col :span="12">
            <el-select 
              v-model="modelRef"
              @change="handleModelChange"
              placeholder="选择模型" 
              size="large" 
              style="margin-bottom: 20px;"
            >
              <el-option label="KNN" value="knn" />
              <el-option label="SVM" value="svm" />
              <el-option label="Random Forest" value="rf" />
              <el-option label="Decision Tree" value="dt" />
            </el-select>
          </el-col>
          <!-- 第二个：任务选择 -->
          <el-col :span="12">
            <el-select 
              v-model="taskRef"
              @change="handleTaskChange"
              placeholder="选择任务" 
              size="large" 
              style="margin-bottom: 20px;"
            >
              <el-option label="classification" value="classification" />
              <el-option label="regression" value="regression" />
            </el-select>
          </el-col>
        </el-row>

        <el-card class="config-panel" shadow="never">
          <el-row :gutter="20">
            <el-col :span="8">
              <div>
                <span style="margin-right: 10px;">表头：</span>
                <el-tag :type="globalState.selectedHeader ? 'info' : 'warning'" size="large" style="font-size: 16px; padding: 6px 12px;">
                  {{ globalState.selectedHeader || '未选择' }}
                </el-tag>
              </div>
            </el-col>
            <el-col :span="8">
              <div>
                <span style="margin-right: 10px;">模型：</span>
                <el-tag :type="globalState.model ? 'success' : 'warning'" size="large" style="font-size: 16px; padding: 6px 12px;">
                  {{ globalState.model.toUpperCase() || '未选择'}}
                </el-tag>
              </div>
            </el-col>
            <el-col :span="8">
              <div>
                <span style="margin-right: 10px;">任务：</span>
                <el-tag :type="globalState.task ? 'primary' : 'warning'" size="large" style="font-size: 16px; padding: 6px 12px;">
                  {{ globalState.task || '未选择' }}
                </el-tag>
              </div>
            </el-col>
          </el-row>
        </el-card>
      </el-main>

      <!-- 右侧 Aside -->
      <el-aside width="120px" style="padding: 20px;">
        <el-button 
          type="primary" 
          @click="startTraining" 
          :loading="loading" 
          style="width: 100%; height: 85%"
          :disabled="!globalState.selectedHeader || !globalState.model || !globalState.task"
        >
          开始训练
        </el-button>
      </el-aside>
    </el-container>
    
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>Query Table</template>
          <Progress :percentage="queryProgress" />
          <div ref="queryChart" class="line-chart"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>Augmented Table</template>
          <Progress :percentage="augProgress" />
          <div ref="augChart" class="line-chart"></div>
        </el-card>
      </el-col>
    </el-row>

</template>

<script setup>
import { ref, inject,computed } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'
import Progress from './progress.vue'

const loading = ref(false)
const queryProgress = ref(0)
const augProgress = ref(0)

const queryChart = ref(null)
const augChart = ref(null)

const queryMetricsHistory = ref([])
const augMetricsHistory = ref([])

let progressSteps = null
let stepIndex = 0
let timer = null

const globalState = inject('globalState')
const setModel = inject('setModel')
const setTask = inject('setTask')

const fileNameRef = computed(() => globalState.value.fileName)
const selectedHeaderRef = computed(() => globalState.value.selectedHeader)
const modelRef = ref(globalState.value.model)
const taskRef = ref(globalState.value.task)

// 处理模型变化
const handleModelChange = (value) => {
  setModel(value)
}
const handleTaskChange = (value) => {
  setTask(value)
}

// 绘制折线图
const drawLineChart = (domRef, metricsList) => {
  const instance = echarts.init(domRef)
  const steps = metricsList.map(m => m.step)

  const seriesNames = ['loss', 'acc', 'pre', 'rec', 'f1']
  const series = seriesNames.map(name => ({
    name,
    type: 'line',
    data: metricsList.map(m => m[name])
  }))

  const option = {
    title: { text: 'Training Metrics', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { top: '10%' },
    xAxis: { type: 'category', data: steps },
    yAxis: { type: 'value' },
    series
  }

  instance.setOption(option)
}

// 点击开始训练按钮，向后端发送开始训练
const startTraining = async () => {
  loading.value = true
  queryProgress.value = 0
  augProgress.value = 0
  stepIndex = 0

  // 构建要发送的命令
  const fileName = fileNameRef.value
  const target = selectedHeaderRef.value
  const model = modelRef.value
  const task = taskRef.value

  const postList = [fileName, target, model, task]
  const command = `python ${fileName} --model ${model} --target "${target}" --task ${task}`

  // 发送 POST 请求
  try {
    await axios.post('http://localhost:3001/training', { postList, command })
    console.log('发送的信息:', postList, '发送的命令:', command)    // 打印发送的命令

    // 获取进度
    const res = await axios.get('http://localhost:3001/progress')

    // 按 type 分组数据
    const allSteps = res.data
    const querySteps = allSteps.filter(step => step.type === 'query')
    const augSteps = allSteps.filter(step => step.type === 'augmented')

    progressSteps = {
      query: querySteps,
      augmented: augSteps
    }

    if (timer) clearInterval(timer)
    timer = setInterval(simulateProgress, 1500)

  } catch (error) {
    console.error('发送请求失败:', error)
    loading.value = false
  }
}

// 模拟获取到的训练进度
const simulateProgress = () => {
  const queryStep = progressSteps.query[stepIndex]
  const augStep = progressSteps.augmented[stepIndex]

  if (queryStep) {
    queryProgress.value = queryStep.progress
    queryMetricsHistory.value.push({
      step: `E${queryStep.epoch}-B${queryStep.batch}`,
      ...queryStep.metrics
    })
    drawLineChart(queryChart.value, queryMetricsHistory.value)
  }

  if (augStep) {
    augProgress.value = augStep.progress
    augMetricsHistory.value.push({
      step: `E${augStep.epoch}-B${augStep.batch}`,
      ...augStep.metrics
    })
    drawLineChart(augChart.value, augMetricsHistory.value)
  }

  stepIndex += 1
  if (stepIndex >= progressSteps.query.length) {
    clearInterval(timer)
    loading.value = false
  }
}

</script>

<style scoped>

.line-chart {
  width: 100%;
  height: 360px;
  background-color: #f9f9f9;
  margin-top: 10px;
}
</style>
