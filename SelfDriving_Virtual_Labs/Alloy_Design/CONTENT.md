
## DENTA

笔记本主要分为以下五个部分：

1.  **第一部分：数据加载与预处理**
    *   导入必要的 Python 库（如 `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`）。
    *   加载合金材料数据集（`data.csv`），其中包含材料标识符 (`sid`)、弹性模量 (`elastic`) 和屈服强度 (`yield`)。
    *   数据预处理：
        *   从材料ID中提取元素成分（Co, Mo, Ti）作为输入特征 (X)。
        *   将弹性模量和屈服强度进行归一化处理，并计算其平均值作为目标性能 (Y)。
        *   可视化数据分布，包括各元素含量的直方图和目标性能值的分布，以及特征之间的散点图矩阵。

2.  **第二部分：定义DANTE算法组件**
    *   此部分旨在集成 DANTE 框架的核心组件。笔记本直接在代码单元格中定义了 DANTE 框架的 Python 类，而不是从外部模块导入。这些类包括：
        *   `SurrogateModel`: 神经网络代理模型的抽象基类。
        *   `DeepActiveLearning`: 实现深度主动学习循环的类。
        *   `ObjectiveFunction`: 目标函数的抽象基类。
        *   `TreeExploration`: 实现基于树的探索策略（如 MCTS）的类。
        *   `Tracker` (在 `utils.py` 中): 用于追踪优化过程的辅助类。
        *   `generate_initial_samples` (在 `utils.py` 中): 生成初始样本点的函数。
    *   定义了一个针对合金优化的特定目标函数类 `AlloyObjectiveFunction`，它继承自 `ObjectiveFunction`。该函数使用现有的实验数据，通过查找最近邻点来评估给定成分的性能，并将问题转化为最小化问题（通过返回负性能值）。

3.  **第三部分：构建神经网络代理模型**
    *   定义了一个 `AlloySurrogateModel` 类，继承自 `SurrogateModel`。
    *   该模型是一个包含批归一化 (Batch Normalization) 和 Dropout 层的全连接神经网络，用于学习合金成分与性能之间的关系。
    *   模型使用 Adam 优化器和均方误差 (MSE) 损失函数进行编译。
    *   使用加载的全部数据集对该代理模型进行训练和初步评估，并可视化其在测试集上的表现。

4.  **第四部分：使用训练的神经网络模型与DANTE进行优化**
    *   定义 `run_dante_optimization` 函数来执行优化流程。
    *   使用在第三部分训练好的 `AlloySurrogateModel` 实例作为 DANTE 优化循环中的代理模型。
    *   创建 `AlloyObjectiveFunction` 的实例。
    *   配置并运行 `DeepActiveLearning` 流程，其中包含一个 `TrainedModelWrapper` 以便直接使用预训练模型，以及一个 `BoundedDeepActiveLearning` 类来确保生成的样本在定义的成分边界内。
    *   优化过程结束后，输出找到的最佳合金成分及其预测性能，并与数据集中最接近的已知材料进行比较。

5.  **第五部分：结果可视化与分析**
    *   定义 `visualize_optimization_results` 函数。
    *   将优化找到的最佳合金成分在原始数据点（成分空间）中进行可视化，包括 3D 散点图和 2D 投影图。
    *   比较优化得到的性能与原始材料性能分布以及数据集中已知的最佳性能。

