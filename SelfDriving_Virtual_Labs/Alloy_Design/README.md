# 合金材料设计优化

本项目使用DANTE框架进行合金材料的成分优化，旨在找到具有最佳机械性能（如弹性模量和屈服强度）的合金成分配比。

## 数据集

`data.csv` - 包含合金材料的成分和性能数据，以下是数据格式：
- `sid`: 材料ID，包含元素成分信息，例如 "Co8.50Mo5.15Ti2.60"
- `elastic`: 弹性模量，单位为Pa
- `yield`: 屈服强度，单位为Pa

## 使用方法

1. 运行Jupyter笔记本进行交互式分析和优化：
   ```
   jupyter notebook DANTE_VL_Alloy_Design.ipynb
   ```

2. 或使用运行脚本执行优化：
   ```
   ./run.sh
   ```

## 文件说明

- `DANTE_VL_Alloy_Design.ipynb`: 用于合金材料优化的Jupyter笔记本
- `data.csv`: 合金材料数据集
- `run.py`: Python运行脚本
- `run.sh`: Shell运行脚本
- `Source_Data/`: 源数据目录（如有必要）

## 优化目标

优化合金成分以最大化弹性模量和屈服强度的综合性能。
