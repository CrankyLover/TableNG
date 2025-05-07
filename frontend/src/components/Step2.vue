<template>
  <!-- ✅ 首屏加载提示 -->
  <div v-if="loadingQueries" class="loading-tip">
    后台正在运行中，请稍后…
  </div>
  <div v-else class="container">
    <h2 class="page-title">索引 CSV 文件预览</h2>

    <!-- 表头选择器 -->
    <div class="header-selector">
      <el-select 
        v-model="selectedHeaderRef"
        @change="handleSelect"
        placeholder="请选择需要分析的列名" 
        size="large" 
        style="width: 100%"
      >
        <el-option
          v-for="header in allHeaders"
          :key="header"
          :label="header"
          :value="header"
        />
      </el-select>
    </div>

    <!-- 表格列表 -->
    <div v-for="(table, index) in tableList" :key="index" class="table-block">
      <div class="table-header" @click="toggleTable(index)">
        <span class="table-title">{{ table.name }}</span>
        <span class="toggle-icon">{{ table.expanded ? '▲ 收起' : '▼ 展开' }}</span>
      </div>

      <div v-if="table.loading" class="loading">正在加载中...</div>

      <div v-if="table.expanded && table.data.length" class="table-wrapper">
        <el-table :data="table.data" border style="min-width: 1000px">
          <el-table-column
            v-for="(col, colIdx) in table.columns"
            :key="colIdx"
            :prop="'col_' + colIdx"
            :label="col"
            header-cell-class-name="header-bold"
          />
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, inject } from 'vue'
import Papa from 'papaparse'
import axios from 'axios'

// const csvFileList = [
//   '../../benchmark/santos_benchmark/query/albums_a.csv',
//   '../../benchmark/santos_benchmark/query/animal_tag_data_a.csv',
//   '../../benchmark/santos_benchmark/query/311_calls_historic_data_a.csv',
//   '../../benchmark/santos_benchmark/query/ipopayments_a.csv',
// ]

const csvFileList = ref([])

const tableList = reactive([])
const allHeaders = reactive([])
const globalState = inject('globalState')
const setSelectedHeader = inject('setSelectedHeader')
const selectedHeaderRef = ref(globalState.value.selectedHeader)
const loadingQueries = ref(true)

// 修改选择时调用
const handleSelect = (value) => {
  setSelectedHeader(value)
  // console.log('当前选择:', value) // 调试用
}

onMounted(() => {
  loadCsvFiles()
})

async function loadCsvFiles() {
  try{
    const { data } = await axios.get('http://localhost:3001/listing')
    csvFileList.value = data.map(item =>  `../../benchmark/santos_benchmark/datalake/${item}`)
  } catch (error) {
    console.error('发送请求失败:', error)
  } finally{
    loadingQueries.value = false
  }
  for (const path of csvFileList.value) {
    const name = path.split('/').pop()
    tableList.push({
      name,
      path,
      expanded: false,
      loading: false,
      columns: [],
      data: []
    })

    try {
      const response = await fetch(path)
      const csvText = await response.text()
      const result = Papa.parse(csvText, {
        preview: 1,
        skipEmptyLines: true
      })
      
      if (result.data.length > 0) {
        const headers = result.data[0].map(h => h.replace(/^\uFEFF/, '')) // 处理BOM字符
        headers.forEach(header => {
          if (!allHeaders.includes(header)) {
            allHeaders.push(header)
          }
        })
      }
    } catch (error) {
      console.error(`加载表头失败：${path}`, error)
    }
  }
}

function toggleTable(index) {
  const table = tableList[index]
  table.expanded = !table.expanded

  if (table.expanded && table.data.length === 0) {
    table.loading = true

    fetch(table.path)
      .then(response => response.text())
      .then(csvText => {
        const result = Papa.parse(csvText, { skipEmptyLines: true })
        const allRows = result.data
        
        if (allRows.length > 0) {
          const headers = allRows[0].map(h => h.replace(/^\uFEFF/, '')) // 处理BOM字符
          table.columns = headers
          
          // 补充添加可能遗漏的表头
          headers.forEach(header => {
            if (!allHeaders.includes(header)) {
              allHeaders.push(header)
            }
          })

          table.data = allRows.slice(1, 21).map(row => {
            const obj = {}
            headers.forEach((col, i) => {
              obj[`col_${i}`] = row[i]
            })
            return obj
          })
        }
        table.loading = false
      })
      .catch(error => {
        console.error(`读取文件 ${table.path} 出错：`, error)
        table.loading = false
      })
  }
}
</script>

<style scoped>
.container {
  padding: 24px;
  font-family: Arial, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
}
.header-selector {
  margin-bottom: 20px;
  font-size: 18px;
}
.page-title {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
}
.upload-wrapper {
  text-align: center;
  margin-bottom: 24px;
}
.table-block {
  margin-bottom: 20px;
  border: 1px solid #ccc;
}
.table-header {
  padding: 12px;
  font-size: 18px;
  font-weight: bold;
  background-color: #f5f5f5;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
}
.table-title {
  flex-grow: 1;
}
.toggle-icon {
  font-size: 14px;
  color: #666;
}
.loading {
  text-align: center;
  font-size: 18px;
  font-weight: bold;
  padding: 20px;
}
.table-wrapper {
  max-height: 500px;
  overflow-x: auto; 
  overflow-y: auto;
  border: 1px solid #ccc;
  scrollbar-gutter: stable both-edges; /* 预留滚动条空间，防止跳动 */
}
.header-bold {
  font-weight: bold;
}
.loading-tip {
  font-size: 20px;
  font-weight: bold;
  text-align: center;
  margin: 80px 0;
  color: #666;
}
</style>