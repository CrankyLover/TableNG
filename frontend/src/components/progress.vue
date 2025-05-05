<template>
    <div class="demo-progress">
      <el-progress
        :percentage="percentage"
        :stroke-width="15"
        striped
        striped-flow
        :status="status"
      />
    </div>
  </template>
  
<script lang="ts" setup>
    defineProps<{
    percentage: number
    status?: 'success' | 'exception' | 'warning' | undefined
    }>()
</script>
  
<style scoped>
.demo-progress .el-progress--line {
    margin-bottom: 15px;
    max-width: 100%;
}
</style>
  