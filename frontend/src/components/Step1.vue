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

    <!-- 上传 / 重新上传 + 确认上传 -->
    <div v-if="uploaded" class="btn-group">
      <!-- 重新上传 -->
      <el-upload
        action=""
        :show-file-list="false"
        :before-upload="handleUpload"
        accept=".csv"
        class="inline-upload" 
      >
        <el-button class="action-btn" type="primary" size="large">
          重新上传
        </el-button>
      </el-upload>

      <!-- 确认上传 -->
      <el-button class="action-btn" type="success" size="large" @click="confirmAndUpload">
        确认上传
      </el-button>
    </div>

    <!-- 初始上传按钮（未选择文件时） -->
    <div v-else class="upload-wrapper">
      <el-upload
        action=""
        :show-file-list="false"
        :before-upload="handleUpload"
        accept=".csv"
      >
        <el-button type="primary" size="large" icon="UploadFilled">
          上传 CSV 文件
        </el-button>
      </el-upload>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import Papa from 'papaparse'
import axios from 'axios'
import { ElMessageBox, ElMessage } from 'element-plus'
import 'element-plus/dist/index.css'

/* ---------------- state ---------------- */
const tableHeader  = ref([])
const tableData    = ref([])
const tableTitle   = ref('')
const loading      = ref(false)
const uploaded     = ref(false)
const selectedFile = ref(null)

/* 限制前 20 行 */
const limitedData = computed(() => tableData.value.slice(0, 20))

/* 表头高度样式 */
const headerStyle = () => ({
  height: '48px',
  'line-height': '48px',
  'font-weight': 'bold'
})

/* ---------------- functions ---------------- */
function handleUpload(file) {
  if (!file.name.endsWith('.csv')) {
    ElMessage.error('仅支持 .csv 格式文件')
    return false
  }

  selectedFile.value = file
  loading.value = true
  tableTitle.value = file.name

  Papa.parse(file, {
    skipEmptyLines: true,
    complete: (res) => {
      const rows = res.data
      if (rows.length < 2) {
        ElMessage.error('文件内容不足')
        loading.value = false
        return
      }
      tableHeader.value = rows[0]
      tableData.value   = rows.slice(1).map(r => {
        const o = {}
        tableHeader.value.forEach((c, i) => { o[`col_${i}`] = r[i] })
        return o
      })
      loading.value = false
      uploaded.value = true
    },
    error: () => {
      ElMessage.error('解析失败，请检查文件格式')
      loading.value = false
    }
  })

  return false
}

function confirmAndUpload() {
  if (!selectedFile.value) {
    ElMessage.warning('当前未选择文件')
    return
  }

  ElMessageBox.confirm(
    '是否确认上传？',
    '确认上传',
    { confirmButtonText: '是', cancelButtonText: '否', type: 'warning' }
  ).then(() => {
    axios.post('http://localhost:3001/sending', { fileName: selectedFile.value.name })
      .then(() => ElMessage.success('上传成功！'))
      .catch(()  => ElMessage.error('上传失败，请重试'))
  }).catch(() => {
    ElMessage.info('已取消上传')
  })
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

.btn-group {
  display: flex;
  justify-content: center;   
  align-items: center;
  gap: 24px;                 
  margin-top: 20px;
}

.inline-upload {
  display: inline-block;
}

.action-btn {
  width: 140px;      
  height: 40px;      
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}

::v-deep(.inline-upload .el-button.action-btn) {
  width: 140px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}
</style>
