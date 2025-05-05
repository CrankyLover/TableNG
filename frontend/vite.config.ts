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
