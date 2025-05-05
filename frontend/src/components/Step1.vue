<template>
  <div class="container">
    <!-- 标题或提示 -->
    <div v-if="loading" class="page-tip">正在加载，请稍候...</div>
    <div v-else-if="!uploaded" class="page-tip">请上传 .csv 格式的文件</div>
    <div v-else class="title">{{ tableTitle }}</div>

    <!-- 表格区域 -->
    <div class="table-wrapper">
      <el-table
          :data="limitedData"
          border
          style="min-width: 1000px"
          scrollbar-always-on
      >
          <el-table-column
              v-for="(col, index) in tableHeader"
              :key="index"
              :label="col"
              :prop="'col_' + index"
              header-cell-class-name="header-bold"
          />
      </el-table>
    </div>

    <!-- 上传按钮 -->
    <div class="upload-wrapper">
      <el-upload
        action=""
        :show-file-list="false"
        :before-upload="handleUpload"
        accept=".csv"
      >
        <el-button type="primary" size="large" :icon="UploadFilled">
          {{ uploaded ? '重新上传' : '上传 CSV 文件' }}
        </el-button>
      </el-upload>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, inject } from 'vue'
import Papa from 'papaparse'
import 'element-plus/dist/index.css'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
  
  // 获取全局状态修改方法
const setFileName = inject('setFileName')  // ✅ 注入方法

const tableHeader = ref([])
const tableData = ref([])
const tableTitle = ref('')
const loading = ref(false)
const uploaded = ref(false)  // 是否已经上传过文件至客户端
const setHeaderStyle = () => {
  return {
    height: '50px',
    'line-height': '50px',
    'font-weight': 'bold'
  }
}
  
// 仅展示前100条
const limitedData = computed(() => tableData.value.slice(0, 20))

// 上传前处理文件
function handleUpload(file) {
  if (!file.name.endsWith('.csv')) {
    ElMessage.error('仅支持 .csv 格式文件')
    return false
  }

  loading.value = true
  tableTitle.value = file.name  // ✅ 使用文件名作为标题

  // ✅ 同步 fileName 到全局状态
  if (setFileName) {
    setFileName(file.name)
  }

  Papa.parse(file, {
    complete: (result) => {
      const rows = result.data.filter((row) => row.length > 0)

      tableHeader.value = rows[0]
      tableData.value = rows.slice(2).map((row) => {
        const obj = {}
        tableHeader.value.forEach((col, index) => {
          obj[`col_${index}`] = row[index]
        })
        return obj
      })

      loading.value = false
      uploaded.value = true  // 成功上传
    },
    error: () => {
      ElMessage.error('解析失败，请检查文件格式')
      loading.value = false
    },
    skipEmptyLines: true
})

return false // 阻止默认上传行为
}
</script>
  
<style scoped>
.container {
  padding: 24px;
  font-family: Arial, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.title {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
}

.page-tip {
  font-size: 20px;
  font-weight: bold;
  text-align: center;
  margin: 60px 0 30px;
  color: #666;
}

.table-wrapper {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #ccc;
  scrollbar-gutter: stable both-edges; /* 预留滚动条空间，防止跳动 */
}

/* 使滚动条更明显：Chrome/Webkit 浏览器 */
.table-wrapper::-webkit-scrollbar {
  height: 12px;
}
.table-wrapper::-webkit-scrollbar-track {
  background: #f0f0f0;
}
.table-wrapper::-webkit-scrollbar-thumb {
  background-color: #888;
  border-radius: 6px;
  border: 2px solid #f0f0f0;
}
.table-wrapper::-webkit-scrollbar-thumb:hover {
  background: #555;
}

.table-scroll {
  overflow-x: auto;
}

.header-bold {
  font-weight: bold !important;
}

.upload-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}

.el-button {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
