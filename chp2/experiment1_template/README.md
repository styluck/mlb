# 实验 1 代码模板

这个模板由课程现有的 `data_io/io_framework.py` 重构而来，保留了“分市场读取—合并—日期筛选—写出”的核心逻辑，并统一了函数名、路径和错误处理。

## 目录

```text
experiment1_template/
├── main.py
├── self_check.py
├── README.md
└── data_io/
    ├── __init__.py
    └── io_framework.py
```

## 完成顺序

1. 打开 `data_io/io_framework.py`，按顺序完成所有 `TODO`。
2. 运行 `python self_check.py`。
3. 修改 `main.py` 中的字段和日期，运行真实数据示例。
4. 确认生成的 CSV 位于 `output/` 目录。

## 命令

```bash
python self_check.py
python main.py
```

`self_check.py` 使用临时数据，因此不需要课程数据集。`main.py` 默认指向仓库现有的：

```text
codes/chp3_data/sample_code/dataset/
```

如果将模板复制到其他位置，请在 `main.py` 中显式修改 `DATA_DIR`。
