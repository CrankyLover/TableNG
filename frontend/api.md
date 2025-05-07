# 端口: http://localhost:3001/

# 操作

## post

### step 3，训练：http://localhost:3001/training

```typescript
// 执行
const postList = [fileName, target, model, task]
const command = `python ${fileName} --model ${model} --target "${target}" --task ${task}`
await axios.post('http://localhost:3001/training', { postList, command })
```

```json
// 格式: {list, string}
  "training": [
    {
      "id": "7e5e",
      "postList": [
        "abandoned_wells_a.csv",
        "track",
        "rf",
        "classification"
      ],
      "command": "python abandoned_wells_a.csv --model rf --target \"track\" --task classification"
    }
  ],
```

## get

### step 3，训练进度：http://localhost:3001/progress

```typescript
// 执行
const res = await axios.get('http://localhost:3001/progress')
```

```json
// 格式: ?
  "progress": [
    {
      "type": "query",
      "epoch": 1,
      "total_epoch": 10,
      "batch": 1,
      "total_batch": 100,
      "progress": 10,
      "metrics": {
        "loss": 0.5,
        "acc": 0.7,
        "pre": 0.68,
        "rec": 0.65,
        "f1": 0.66
      },
      "confusion_matrix": [
        [
          40,
          10
        ],
        [
          8,
          42
        ]
      ],
      "id": "109c"
    },
```

