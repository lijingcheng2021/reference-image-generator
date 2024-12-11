# 参照图数据生成工具

一个用于生成工地安全参照图对的自动化工具，基于通义千问VL大模型进行图片分析和匹配。

## 功能特点

- 自动分析工地场景图片中的设备和人员信息
- 智能匹配相似场景的图片对
- 生成标准化的参照图数据集
- 支持进度显示和错误处理

## 项目结构

```
.
├── src/
│   ├── main.py          # 主程序入口
│   ├── api_client.py    # API客户端封装
│   └── analyzer.py      # 图片分析器
├── data/
│   ├── images/          # 原始图片目录
│   └── reference_dataset/  # 生成的数据集
```

## 安装依赖

```bash
pip install openai tqdm json-repair
```

## 使用方法

1. 配置API参数

```python
client = ModelClient(
    api_key='YOUR_API_KEY',
    base_url='YOUR_API_BASE_URL'
)
```

2. 准备图片数据
- 将待分析的图片放入 `data/images` 目录
- 支持的图片格式：jpg, jpeg, png

3. 运行程序

```bash
python src/main.py
```

## 输出格式

生成的数据集将保存在 `data/reference_dataset` 目录下���包含：
- `ref_*.jpg`：参照图片
- `test_*.jpg`：测试图片
- `pair_*.json`：图片对的分析信息

JSON文件格式示例：

```json
{
  "reference": {
    "analysis": {
      "devices": ["设备描述1", "设备描述2"],
      "personnel": ["人员描述1", "人员描述2"]
    },
    "image_path": "图片路径"
  },
  "test": {
    // 同上
  }
}
```

## 注意事项

1. 确保图片目录具有正确的读写权限
2. 建议每批处理的图片数量不要过多，以免API调用超时
3. 程序会自动过滤掉无法匹配的图片对

## 错误处理

- 程序会跳过分析失败的图片
- 无法匹配的图片会被自动过滤
- 所有错误都会被记录并显示在控制台

## 开发者

李某某

## 许可证

MIT License
