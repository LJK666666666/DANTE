# 🚀 DANTE Alloy Design - Quick Start Guide

## 🎯 Best Way to Use: Interactive Jupyter Notebook

我们强烈推荐使用 **main.ipynb** 来运行DANTE合金设计工作流程，因为它结合了模块化代码和交互式可视化的优势。

### ✨ 为什么选择 main.ipynb？

1. **🎯 交互式体验**: 逐步执行，实时查看结果
2. **📊 内联可视化**: 图表直接显示在notebook中
3. **🔍 实时分析**: 立即获得反馈和结果
4. **📝 内置文档**: 详细的说明和指导
5. **🛠️ 易于调试**: 方便修改和实验
6. **💡 教育性强**: 完美的学习和探索工具
7. **🏗️ 模块化**: 核心功能在.py文件中，保持代码整洁
8. **🧠 智能加载**: 自动加载原始notebook训练的模型权重

## 🚀 快速开始

### 方法1: 自动启动脚本（推荐）

#### Windows用户:
```bash
# 双击运行或在命令行中执行
start_notebook.bat
```

#### Linux/Mac用户:
```bash
# 在终端中执行
./start_notebook.sh
```

#### 或者使用Python脚本:
```bash
python start_notebook.py
```

### 方法2: 手动启动

```bash
# 1. 安装Jupyter (如果还没有)
pip install jupyter

# 2. 启动notebook
jupyter notebook main.ipynb
```

## 📋 系统要求

### 必需依赖
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### 可选依赖（用于高级功能）
- tensorflow/keras (神经网络)
- dante框架 (高级优化)
- jupyter (交互式notebook)

### 安装依赖
```bash
# 安装基本依赖
pip install -r requirements.txt

# 安装可选依赖
pip install tensorflow jupyter
```

## 📁 文件结构

```
src/
├── main.ipynb              # 🌟 主要的交互式notebook
├── main.py                 # 命令行版本
├── start_notebook.py       # 自动启动脚本
├── start_notebook.bat      # Windows启动脚本
├── start_notebook.sh       # Linux/Mac启动脚本
├── data_loader.py          # 数据加载模块
├── alloy_objective.py      # 目标函数模块
├── neural_models.py        # 神经网络模块
├── optimization.py         # 优化算法模块
├── visualization.py        # 可视化模块
├── config.py              # 配置管理
├── run_example.py         # 示例脚本
├── test_imports.py        # 模块测试
└── requirements.txt       # 依赖列表
```

## 🎮 使用流程

### 在 main.ipynb 中:

1. **📦 导入和设置** - 导入所有必要的模块
2. **📊 数据加载** - 加载和预处理合金数据
3. **🎯 目标函数** - 创建优化目标函数
4. **🧠 模型训练** - 训练神经网络代理模型
5. **🚀 优化** - 运行DANTE优化算法
6. **📈 可视化** - 生成综合分析图表
7. **📋 结果总结** - 查看优化结果和性能指标

### 每个步骤都可以:
- ⏯️ 单独执行
- 🔧 修改参数
- 📊 查看中间结果
- 🐛 调试问题

## 💡 使用技巧

### Jupyter Notebook 技巧:
- `Shift + Enter`: 运行当前单元格
- `Ctrl + Enter`: 运行当前单元格但不移动到下一个
- `A`: 在上方插入新单元格
- `B`: 在下方插入新单元格
- `DD`: 删除当前单元格
- `M`: 将单元格转换为Markdown
- `Y`: 将单元格转换为代码

### 实验建议:
1. **🔬 参数调整**: 修改优化参数，观察结果变化
2. **📊 数据探索**: 在数据加载部分添加自己的分析
3. **🎨 可视化定制**: 修改图表样式和内容
4. **🧪 成分测试**: 在交互分析部分测试不同的合金成分

## 🧠 模型权重管理

### 智能加载机制:

系统会自动处理模型权重的加载和保存：

1. **🔍 自动检测**: 首先检查原始模型权重目录 `../model_weights/`
2. **📥 智能加载**: 如果找到预训练权重，自动加载并验证
3. **🚀 快速启动**: 使用预训练模型，跳过训练过程
4. **🔄 备用训练**: 如果权重不存在或加载失败，自动训练新模型
5. **💾 双重保存**: 新训练的模型保存到本地 `model_weights/` 目录

### 权重文件说明:

```
../model_weights/                           # 原始预训练权重
├── dual_network_elastic.weights.h5        # 弹性模量预测模型
├── dual_network_yield.weights.h5          # 屈服强度预测模型
├── dual_network_scalers.pkl               # 双网络数据缩放器
├── improved_phase_composition_final.weights.h5  # 相组成预测模型
└── improved_phase_composition_scalers.pkl # 相组成数据缩放器

src/model_weights/                          # 新训练的权重
├── (新训练的模型权重会保存在这里)
└── (保持原始权重不被覆盖)
```

### 优势:

- ⚡ **快速启动**: 直接使用预训练模型，无需等待训练
- 🔒 **权重保护**: 原始权重不会被意外覆盖
- 🎯 **智能回退**: 自动处理各种异常情况
- 📊 **性能保证**: 使用经过验证的高质量模型

## 🔧 故障排除

### 常见问题:

1. **Jupyter未安装**:
   ```bash
   pip install jupyter
   ```

2. **TensorFlow缺失**:
   ```bash
   pip install tensorflow
   ```
   或者使用fallback模型（自动处理）

3. **DANTE框架缺失**:
   - 系统会自动使用简单优化算法
   - 不影响主要功能

4. **数据文件未找到**:
   - 系统会自动创建合成数据进行演示
   - 或者将您的数据文件放在正确位置

5. **内存不足**:
   - 减少batch_size
   - 减少n_folds
   - 使用更简单的模型

## 📈 输出文件

运行完成后，您将获得:

### 📊 可视化图表 (figures/ 目录):
- `element_distributions.png` - 元素成分分布
- `property_distributions.png` - 机械性能分布
- `model_performance.png` - 模型预测精度
- `3d_composition_space.png` - 3D成分空间
- `optimization_convergence.png` - 优化收敛过程
- 等等...

### 🧠 模型文件 (model_weights/ 目录):
- 训练好的神经网络权重
- 数据预处理器
- 模型配置文件

### 📋 结果数据:
- 最优合金成分
- 性能预测值
- 优化历史记录

## 🎉 下一步

1. **📊 分析结果**: 查看生成的可视化图表
2. **🔬 实验验证**: 用实验数据验证预测结果
3. **🎛️ 参数调优**: 尝试不同的优化参数
4. **🚀 扩展应用**: 将框架应用到您的具体问题

## 📞 获取帮助

如果遇到问题:
1. 检查错误信息和建议
2. 查看notebook中的详细文档
3. 运行 `python test_imports.py` 检查模块状态
4. 查看生成的日志文件

---

**🎯 记住**: main.ipynb 是最佳的使用方式，它结合了专业代码结构和交互式体验的优势！
