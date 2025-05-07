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
              <el-option label="SVM" value="svm" />
              <el-option label="Gradient Boosting" value="gradient_boosting" />
              <el-option label="Random Forest" value="random_forest" />
              <el-option label="Decision Tree" value="decision_tree" />
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
import { ref, inject, computed, onMounted, onBeforeUnmount } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'
import Progress from './progress.vue'

// 响应式状态
const loading = ref(false)
const queryProgress = ref(0)
const augProgress = ref(0)

// 图表引用
const queryChart = ref(null)
const augChart = ref(null)

// 指标数据存储结构
const metricsData = ref({
  query: {
    train: [],
    test: []
  },
  augmented: {
    train: [],
    test: []
  }
})

// 图表实例和观察器
let queryChartInstance = null
let augChartInstance = null
let resizeObserver = null

// 轮询控制
let polling = true

// 全局状态
const globalState = inject('globalState')
const setModel = inject('setModel')
const setTask = inject('setTask')

// 计算属性
const fileNameRef = computed(() => globalState.value.fileName)
const selectedHeaderRef = computed(() => globalState.value.selectedHeader)
const modelRef = ref(globalState.value.model)
const taskRef = ref(globalState.value.task)

// 图表配置
const chartOptions = {
  title: { text: 'Training Metrics', left: 'center' },
  tooltip: { trigger: 'axis' },
  legend: { 
    top: '10%',
    // data: ['train-loss', 'train-acc', 'train-f1', 'test-loss', 'test-acc', 'test-f1']
    data: ['train-loss', 'train-acc', 'test-loss', 'test-acc']
  },
  xAxis: { type: 'category' },
  yAxis: { type: 'value' },
  series: [
    { name: 'train-loss', type: 'line', lineStyle: { type: 'dashed' } },
    { name: 'train-acc', type: 'line', lineStyle: { type: 'dashed' } },
    // { name: 'train-f1', type: 'line', lineStyle: { type: 'dashed' } },
    { name: 'test-loss', type: 'line' },
    { name: 'test-acc', type: 'line' },
    // { name: 'test-f1', type: 'line' }
  ]
}

// 初始化图表
const initChart = (dom) => {
  const chart = echarts.init(dom)
  chart.setOption(chartOptions)
  return chart
}

// 更新图表数据
const updateChart = (chart, typeData) => {
  const allSteps = [
    ...typeData.train.map(m => m.step),
    ...typeData.test.map(m => m.step)
  ].filter((v, i, a) => a.indexOf(v) === i) // 去重

  const option = {
    xAxis: { data: allSteps },
    series: [
      { data: typeData.train.map(m => m.loss) },
      { data: typeData.train.map(m => m.acc) },
      // { data: typeData.train.map(m => m.f1) },
      { data: typeData.test.map(m => m.loss) },
      { data: typeData.test.map(m => m.acc) },
      // { data: typeData.test.map(m => m.f1) }
    ]
  }
  
  chart.setOption({
    ...option,
    series: option.series.map((data, index) => ({
      ...chart.getOption().series[index],
      data: data.data
    }))
  })
}

// 处理进度数据
const processProgress = (progress) => {
  // 先分类
  const grouped = { query: 0, augmented: 0 }

  progress.forEach(item => {
    const step = `E${item.epoch}`
    const targetList = metricsData.value[item.type][item.mode]

    if (!targetList.some(m => m.step === step)) {
      const metrics = {
        step,
        loss: item.metrics.loss,
        acc: item.metrics.acc,
      }
      targetList.push(metrics)
    }

    grouped[item.type]++
  })

  // 分别计算进度
  queryProgress.value = Math.min((grouped.query / 20) * 100, 100)
  augProgress.value = Math.min((grouped.augmented / 20) * 100, 100)
}

// 轮询训练进度
const pollProgress = async () => {
  try {
    while (polling) {
      const res = await axios.get('http://localhost:3001/progress')
      console.log(res, polling)
      const isFinished = res.data.finish
      const progressList = res.data.progress;
      console.log(isFinished, progressList.length)
      
      if (isFinished || progressList.length >= 40) {
        polling = false
        loading.value = false
      }

      console.log(progressList)
      // 分别处理 query / augmented 数据(增量，非全量替换，缩小重新绘制时间，以及刷新体验差)
      processProgress(progressList)

      // 重新绘制图表
      if (queryChartInstance) {
        updateChart(queryChartInstance, metricsData.value.query)
      }
      if (augChartInstance) {
        updateChart(augChartInstance, metricsData.value.augmented)
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
  } catch (error) {
    console.error('获取进度失败:', error)
    polling = false
    loading.value = false
  }
}

// 开始训练
const startTraining = async () => {
  try {
    loading.value = true
    polling = true
    
    // 重置数据
    metricsData.value = {
      query: { train: [], test: [] },
      augmented: { train: [], test: [] }
    }

    // 发送训练请求
    await axios.post('http://localhost:3001/training', {
      postList: [
        fileNameRef.value,
        selectedHeaderRef.value,
        modelRef.value,
        taskRef.value
      ],
      command: `python ${fileNameRef.value} --model ${modelRef.value} --target "${selectedHeaderRef.value}" --task ${taskRef.value}`
    })

    // 开始轮询
    pollProgress()
  } catch (error) {
    console.error('启动训练失败:', error)
    loading.value = false
    polling = false
  }
}

// 处理模型变化
const handleModelChange = (value) => {
  setModel(value)
}

// 处理任务变化
const handleTaskChange = (value) => {
  setTask(value)
}

// 生命周期钩子
onMounted(() => {
  // 初始化图表
  queryChartInstance = initChart(queryChart.value)
  augChartInstance = initChart(augChart.value)

  // 自动调整图表大小
  resizeObserver = new ResizeObserver(() => {
    queryChartInstance?.resize()
    augChartInstance?.resize()
  })
  
  resizeObserver.observe(queryChart.value)
  resizeObserver.observe(augChart.value)
})

onBeforeUnmount(() => {
  polling = false
  resizeObserver?.disconnect()
  queryChartInstance?.dispose()
  augChartInstance?.dispose()
})
</script>

<style scoped>

.line-chart {
  width: 100%;
  height: 360px;
  background-color: #f9f9f9;
  margin-top: 10px;
}
</style>
