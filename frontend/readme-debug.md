# 1、页面开发

把组件放到component里面，用import引用

注意需要安装的包
```shell
npm install element-plus --save
npm install axios --save
npm install echarts --save
npm install json-server --save
npm install papaparse --save
```

# 2、虚拟调试

使用json-server模拟发送接收消息

创建./json-server/db.json放置模拟数据

在package.json添加
```json
  "scripts": {
    "json-server": "json-server --watch ./json-server/db.json --port 3001"
  },
```

创建两个terminal，进入frontend，分别执行
```shell
npm run json-server
```
```shell
npm run dev
```

# 3、修改step2

希望加入本地读取，但是面临一个问题：./frontend里面的server无法读取./benchmark里的文件，使用相对路径../访问不会有权限(因为安全性考虑)
在这里，gpt给出两种解决方案
1. 一个是copy到./frontend/public/里，这样很明显会有冗余内存占用
1. 一个是直接暴露给server，会有一定的安全风险，但是本项目不需要考虑那么多
因此选择第二种方案，直接暴露父目录下的其他内容。这样就需要修改./frontend/vite.config.ts内容
```typesrcipt
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

import path from 'path' // 加载路径，使得能够访问前面的文件夹（csv所在）
import fs from 'fs'
const benchmarkPath = path.resolve(__dirname, '../benchmark')

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
    {
      name: 'serve-csv-from-benchmark',
      configureServer(server) {
        server.middlewares.use('/benchmark', (req, res, next) => {
          const filePath = path.join(benchmarkPath, req.url!.replace(/^\/benchmark/, ''))
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'text/csv')
            fs.createReadStream(filePath).pipe(res)
          } else {
            res.statusCode = 404
            res.end('File not found')
          }
        })
      }
    }
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    fs: {
      allow: [
        benchmarkPath,     // 允许访问 benchmark 文件夹
        path.resolve(__dirname),              // 当前项目 frontend 根目录
        path.resolve(__dirname, 'node_modules') // node_modules 也要显式允许
      ]
    }
  }
})
```