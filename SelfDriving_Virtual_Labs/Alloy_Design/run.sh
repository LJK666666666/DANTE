#!/bin/bash

# 合金材料设计优化运行脚本

echo "开始合金材料设计优化..."

# 检查数据集是否存在
if [ ! -f "data.csv" ]; then
    echo "错误：数据集文件data.csv不存在！"
    exit 1
fi

# 运行Python脚本
echo "运行DANTE优化算法..."
python run.py

echo "优化完成！"
