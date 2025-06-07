#!/usr/bin/env python3
"""
DANTE迭代优化系统执行脚本
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 设置随机种子以确保可重现性
np.random.seed(42)
tf.random.set_seed(42)

# 设置可视化样式
plt.style.use('ggplot')
sns.set(style="whitegrid")

print("=" * 80)
print("DANTE迭代优化系统：科学计算反馈闭环")
print("=" * 80)
print(f"TensorFlow版本: {tf.__version__}")
print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 添加DANTE模块到路径
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

# 导入DANTE模块
try:
    from dante.neural_surrogate import SurrogateModel
    from dante.deep_active_learning import DeepActiveLearning
    from dante.obj_functions import ObjectiveFunction
    from dante.tree_exploration import TreeExploration
    from dante.utils import generate_initial_samples, Tracker
    print("✓ 成功导入DANTE模块！")
except ImportError as e:
    print(f"✗ 导入DANTE模块失败: {e}")
    sys.exit(1)

# 加载初始数据
data_path = "data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✓ 成功加载初始数据集，共 {len(df)} 个样本")
else:
    print("✗ 数据文件不存在！")
    sys.exit(1)

# 成分提取函数
def extract_composition(sid):
    """从材料ID中提取元素成分"""
    elements = ['Co', 'Mo', 'Ti']
    values = []
    
    for element in elements:
        if element in sid:
            pos = sid.find(element) + len(element)
            next_pos = len(sid)
            for next_elem in elements:
                if next_elem != element and sid.find(next_elem, pos) != -1:
                    next_pos = min(next_pos, sid.find(next_elem, pos))
            value = float(sid[pos:next_pos])
            values.append(value)
        else:
            values.append(0.0)
    
    co, mo, ti = values
    fe = 100.0 - co - mo - ti
    return [co, mo, ti, fe]

# 预处理初始数据
composition_values = df['sid'].apply(extract_composition)
X_initial = np.array(composition_values.tolist())

# 提取并标准化目标值
elastic_values = df['elastic'].values
yield_values = df['yield'].values

elastic_min, elastic_max = np.min(elastic_values), np.max(elastic_values)
yield_min, yield_max = np.min(yield_values), np.max(yield_values)

elastic_norm = (elastic_values - elastic_min) / (elastic_max - elastic_min)
yield_norm = (yield_values - yield_min) / (yield_max - yield_min)
Y_initial = (elastic_norm + yield_norm) / 2

print(f"✓ 初始数据预处理完成：{X_initial.shape[0]} 个样本，{X_initial.shape[1]} 个特征")
print(f"成分范围检查：Co[{X_initial[:, 0].min():.2f}, {X_initial[:, 0].max():.2f}], "
      f"Mo[{X_initial[:, 1].min():.2f}, {X_initial[:, 1].max():.2f}], "
      f"Ti[{X_initial[:, 2].min():.2f}, {X_initial[:, 2].max():.2f}], "
      f"Fe[{X_initial[:, 3].min():.2f}, {X_initial[:, 3].max():.2f}]")

# 科学计算模拟器
class ScientificComputationSimulator:
    """科学计算模拟器，模拟真实的DFT/MD计算过程"""
    
    def __init__(self, base_data_X, base_data_Y, noise_level=0.05, computation_time=0.5):
        self.base_X = base_data_X
        self.base_Y = base_data_Y
        self.noise_level = noise_level
        self.computation_time = computation_time
        self.calculation_count = 0
        self.calculation_history = []
        
    def validate_composition(self, composition):
        """验证成分的物理合理性"""
        composition = np.array(composition)  # 确保是numpy数组
        
        if np.any(composition < 0):
            return False, "成分不能为负数"
        
        total = np.sum(composition)
        if not np.isclose(total, 100.0, rtol=0.05):
            return False, f"成分总和不等于100%: {total:.2f}%"
        
        co, mo, ti, fe = composition
        if co > 50 or mo > 30 or ti > 20 or fe < 30:
            return False, "成分超出合理范围"
        
        return True, "成分验证通过"
    
    def compute_properties(self, composition):
        """模拟科学计算过程"""
        self.calculation_count += 1
        start_time = time.time()
        
        composition = np.array(composition)  # 确保是numpy数组
        
        print(f"  [科学计算 #{self.calculation_count}] 成分: "
              f"Co={composition[0]:.2f}%, Mo={composition[1]:.2f}%, "
              f"Ti={composition[2]:.2f}%, Fe={composition[3]:.2f}%")
        
        # 验证成分
        is_valid, message = self.validate_composition(composition)
        if not is_valid:
            print(f"  [失败] {message}")
            return None, None, False
        
        # 模拟计算时间
        time.sleep(self.computation_time)
        
        # 使用最近邻插值加噪声来模拟真实计算
        distances = np.linalg.norm(self.base_X - composition, axis=1)
        nearest_indices = np.argsort(distances)[:3]
        
        weights = 1.0 / (distances[nearest_indices] + 1e-6)
        weights = weights / np.sum(weights)
        
        base_performance = np.sum(weights * self.base_Y[nearest_indices])
        
        # 添加计算噪声
        noise = np.random.normal(0, self.noise_level)
        final_performance = base_performance + noise
        
        computation_time = time.time() - start_time
        
        # 记录计算历史
        record = {
            'composition': composition.copy(),
            'performance': final_performance,
            'computation_time': computation_time,
            'timestamp': datetime.now().isoformat()
        }
        self.calculation_history.append(record)
        
        print(f"  [成功] 性能: {final_performance:.6f} (用时: {computation_time:.2f}s)")
        
        return final_performance, computation_time, True
    
    def get_calculation_summary(self):
        """获取计算统计摘要"""
        if not self.calculation_history:
            return "尚未进行任何计算"
        
        performances = [r['performance'] for r in self.calculation_history]
        times = [r['computation_time'] for r in self.calculation_history]
        
        summary = f"""
科学计算统计摘要:
- 总计算次数: {self.calculation_count}
- 平均计算时间: {np.mean(times):.2f} ± {np.std(times):.2f} 秒
- 性能值范围: [{np.min(performances):.6f}, {np.max(performances):.6f}]
- 平均性能: {np.mean(performances):.6f} ± {np.std(performances):.6f}
        """
        return summary

# 创建科学计算模拟器
sci_computer = ScientificComputationSimulator(X_initial, Y_initial, noise_level=0.03, computation_time=0.5)
print("✓ 科学计算模拟器创建完成")

# 测试科学计算模拟器
test_composition = [25.0, 15.0, 10.0, 50.0]
print(f"\n测试科学计算模拟器...")
performance, comp_time, success = sci_computer.compute_properties(test_composition)
if success:
    print(f"✓ 模拟器测试成功")
else:
    print("✗ 模拟器测试失败")

print(f"\n继续初始化系统组件...")

# 自适应代理模型
class AdaptiveSurrogateModel:
    """自适应代理模型，支持增量学习"""
    
    def __init__(self, input_dim=4, initial_data=None):
        self.input_dim = input_dim
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.model_version = 0
        self.training_history = []
        
        if initial_data is not None:
            X_train, Y_train = initial_data
            self.initialize_model(X_train, Y_train)
    
    def create_neural_network(self):
        """创建神经网络架构"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def initialize_model(self, X_train, Y_train):
        """初始化模型"""
        self.model = self.create_neural_network()
        
        # 标准化数据
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
        
        # 训练模型
        X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
            X_scaled, Y_scaled, test_size=0.2, random_state=42
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train_split, Y_train_split,
            validation_data=(X_val_split, Y_val_split),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.model_version += 1
        
        # 评估模型
        val_predictions = self.model.predict(X_val_split, verbose=0)
        val_mse = mean_squared_error(Y_val_split, val_predictions.flatten())
        val_r2 = r2_score(Y_val_split, val_predictions.flatten())
        
        training_record = {
            'version': self.model_version,
            'training_samples': len(X_train),
            'validation_mse': val_mse,
            'validation_r2': val_r2,
            'epochs_trained': len(history.history['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_record)
        
        print(f"✓ 初始模型训练完成 (v{self.model_version})")
        print(f"  训练样本: {len(X_train)}, 验证MSE: {val_mse:.6f}, 验证R²: {val_r2:.3f}")
        
        return history
    
    def update_model(self, new_X, new_Y, epochs=50):
        """增量更新模型"""
        if self.model is None:
            raise ValueError("模型尚未初始化")
        
        # 标准化新数据
        new_X_scaled = self.scaler_X.transform(new_X)
        new_Y_scaled = self.scaler_Y.transform(new_Y.reshape(-1, 1)).flatten()
        
        # 增量训练
        history = self.model.fit(
            new_X_scaled, new_Y_scaled,
            epochs=epochs,
            batch_size=min(32, len(new_X)),
            verbose=0
        )
        
        self.model_version += 1
        
        # 评估更新后的模型
        predictions = self.model.predict(new_X_scaled, verbose=0)
        mse = mean_squared_error(new_Y_scaled, predictions.flatten())
        r2 = r2_score(new_Y_scaled, predictions.flatten())
        
        training_record = {
            'version': self.model_version,
            'new_samples': len(new_X),
            'incremental_mse': mse,
            'incremental_r2': r2,
            'epochs_trained': epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_record)
        
        print(f"✓ 模型增量更新完成 (v{self.model_version})")
        print(f"  新样本: {len(new_X)}, 增量MSE: {mse:.6f}, 增量R²: {r2:.3f}")
        
        return history
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未初始化")
        
        X_scaled = self.scaler_X.transform(X)
        Y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        
        return Y_pred.flatten()
    
    def save_model(self, filename_prefix):
        """保存模型"""
        if self.model is None:
            return
        
        filename = f"{filename_prefix}_v{self.model_version}.keras"
        self.model.save(filename)
        print(f"✓ 模型已保存: {filename}")
    
    def get_training_summary(self):
        """获取训练历史摘要"""
        if not self.training_history:
            return "尚未进行模型训练"
        
        summary = "模型训练历史:\n"
        for record in self.training_history:
            summary += f"  v{record['version']}: "
            if 'training_samples' in record:
                summary += f"初始训练 {record['training_samples']} 样本, R²={record['validation_r2']:.3f}\n"
            else:
                summary += f"增量学习 {record['new_samples']} 样本, R²={record['incremental_r2']:.3f}\n"
        
        return summary

# 初始化自适应代理模型
adaptive_model = AdaptiveSurrogateModel(input_dim=4, initial_data=(X_initial, Y_initial))

# 迭代DANTE优化器
class IterativeDANTEOptimizer:
    """迭代DANTE优化器，集成科学计算反馈"""
    
    def __init__(self, initial_X, initial_Y, scientific_computer, surrogate_model):
        self.X_data = initial_X.copy()
        self.Y_data = initial_Y.copy()
        self.sci_computer = scientific_computer
        self.surrogate_model = surrogate_model
        self.iteration_count = 0
        self.optimization_history = []
    
    def enforce_composition_constraints(self, composition):
        """强制满足成分约束"""
        # 确保非负
        composition = np.maximum(composition, 0.0)
        
        # 归一化到100%
        total = np.sum(composition)
        if total > 0:
            composition = composition * (100.0 / total)
        
        # 检查物理约束
        co, mo, ti, fe = composition
        
        # 软约束：如果超出合理范围，进行调整
        if co > 50:
            excess = co - 50
            composition[0] = 50
            composition[3] += excess  # 多余的Co转为Fe
        
        if mo > 30:
            excess = mo - 30
            composition[1] = 30
            composition[3] += excess
        
        if ti > 20:
            excess = ti - 20
            composition[2] = 20
            composition[3] += excess
        
        # 重新归一化
        total = np.sum(composition)
        composition = composition * (100.0 / total)
        
        return composition
    
    def generate_candidates(self, num_candidates=5):
        """生成候选解"""
        print(f"  生成 {num_candidates} 个候选解...")
        
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 3
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            
            # 使用当前最佳解附近的随机扰动
            best_idx = np.argmax(self.Y_data)
            best_composition = self.X_data[best_idx]
            
            # 随机扰动
            perturbation = np.random.normal(0, 5.0, 4)  # 5%的标准差
            candidate = best_composition + perturbation
            
            # 应用约束
            candidate = self.enforce_composition_constraints(candidate)
            
            # 检查是否太接近现有点
            min_distance = np.min(np.linalg.norm(self.X_data - candidate, axis=1))
            if min_distance > 2.0:  # 至少2%的距离
                candidates.append(candidate)
        
        # 如果候选解不够，用纯随机生成补充
        while len(candidates) < num_candidates:
            random_comp = np.random.dirichlet([1, 1, 1, 1]) * 100
            random_comp = self.enforce_composition_constraints(random_comp)
            candidates.append(random_comp)
        
        print(f"  ✓ 生成 {len(candidates)} 个候选解")
        return np.array(candidates)
    
    def run_iteration(self, num_candidates=3):
        """运行一次完整的迭代"""
        self.iteration_count += 1
        iteration_start_time = time.time()
        
        print(f"\n迭代 #{self.iteration_count} 开始")
        print(f"当前数据集大小: {len(self.X_data)} 个样本")
        
        # 生成候选解
        candidates = self.generate_candidates(num_candidates)
        
        # 科学计算验证
        print(f"  开始科学计算验证...")
        verified_results = []
        successful_calculations = 0
        
        for i, candidate in enumerate(candidates):
            performance, comp_time, success = self.sci_computer.compute_properties(candidate)
            
            if success:
                verified_results.append({
                    'composition': candidate,
                    'performance': performance,
                    'computation_time': comp_time
                })
                successful_calculations += 1
        
        # 更新数据集
        if verified_results:
            new_X = np.array([r['composition'] for r in verified_results])
            new_Y = np.array([r['performance'] for r in verified_results])
            
            self.X_data = np.vstack([self.X_data, new_X])
            self.Y_data = np.hstack([self.Y_data, new_Y])
            
            # 增量更新代理模型
            print(f"  更新代理模型...")
            self.surrogate_model.update_model(new_X, new_Y)
        
        # 记录迭代历史
        iteration_time = time.time() - iteration_start_time
        best_idx = np.argmax(self.Y_data)
        
        iteration_record = {
            'iteration': self.iteration_count,
            'candidates_generated': len(candidates),
            'successful_calculations': successful_calculations,
            'new_samples_added': len(verified_results),
            'total_samples': len(self.X_data),
            'best_performance': self.Y_data[best_idx],
            'best_composition': self.X_data[best_idx].copy(),
            'iteration_time': iteration_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(iteration_record)
        
        print(f"迭代 #{self.iteration_count} 完成 (用时: {iteration_time:.2f}s)")
        print(f"成功计算: {successful_calculations}/{len(candidates)}")
        print(f"当前最佳性能: {iteration_record['best_performance']:.6f}")
        print(f"最佳成分: Co={iteration_record['best_composition'][0]:.2f}%, "
              f"Mo={iteration_record['best_composition'][1]:.2f}%, "
              f"Ti={iteration_record['best_composition'][2]:.2f}%, "
              f"Fe={iteration_record['best_composition'][3]:.2f}%")
        
        return iteration_record
    
    def get_current_best(self):
        """获取当前最佳结果"""
        best_idx = np.argmax(self.Y_data)
        return {
            'composition': self.X_data[best_idx],
            'performance': self.Y_data[best_idx],
            'index': best_idx
        }

# 创建迭代优化器
dante_optimizer = IterativeDANTEOptimizer(X_initial, Y_initial, sci_computer, adaptive_model)
print("✓ 迭代DANTE优化器创建完成")

print(f"\n" + "="*80)
print("开始迭代优化过程")
print(f"="*80)

# 运行多轮迭代优化
num_iterations = 5
candidates_per_iteration = 3

print(f"计划进行 {num_iterations} 轮迭代，每轮 {candidates_per_iteration} 个候选解")

# 记录初始状态
initial_best = dante_optimizer.get_current_best()
print(f"初始最佳性能: {initial_best['performance']:.6f}")
print(f"初始最佳成分: Co={initial_best['composition'][0]:.2f}%, "
      f"Mo={initial_best['composition'][1]:.2f}%, "
      f"Ti={initial_best['composition'][2]:.2f}%, "
      f"Fe={initial_best['composition'][3]:.2f}%")

# 执行迭代
try:
    for iteration in range(num_iterations):
        print(f"\n" + "="*60)
        print(f"执行迭代 {iteration + 1}/{num_iterations}")
        print(f"="*60)
        
        # 运行一次完整迭代
        iteration_result = dante_optimizer.run_iteration(
            num_candidates=candidates_per_iteration
        )
        
        # 显示进展
        if iteration > 0:
            prev_best = dante_optimizer.optimization_history[iteration-1]['best_performance']
            current_best = iteration_result['best_performance']
            improvement = current_best - prev_best
            print(f"性能改进: {improvement:+.6f}")
        
        # 中途保存（可选）
        if (iteration + 1) % 2 == 0:
            adaptive_model.save_model(f"adaptive_model_iter_{iteration + 1}")

except KeyboardInterrupt:
    print("\n优化过程被用户中断")
except Exception as e:
    print(f"\n优化过程出现错误: {e}")
    import traceback
    traceback.print_exc()

# 显示最终结果
final_best = dante_optimizer.get_current_best()
print(f"\n" + "="*80)
print("迭代优化完成！")
print(f"="*80)
print(f"最终最佳性能: {final_best['performance']:.6f}")
print(f"最终最佳成分: Co={final_best['composition'][0]:.2f}%, "
      f"Mo={final_best['composition'][1]:.2f}%, "
      f"Ti={final_best['composition'][2]:.2f}%, "
      f"Fe={final_best['composition'][3]:.2f}%")

if 'initial_best' in locals():
    total_improvement = final_best['performance'] - initial_best['performance']
    print(f"总体改进: {total_improvement:+.6f} ({total_improvement/initial_best['performance']*100:+.2f}%)")

print(f"总计算次数: {sci_computer.calculation_count}")
print(f"数据集最终大小: {len(dante_optimizer.X_data)} 个样本")
print(f"模型当前版本: {adaptive_model.model_version}")

# 生成优化报告
print(f"\n" + "="*80)
print("优化报告")
print(f"="*80)

print(f"\n【优化统计】")
print(f"初始数据集大小: {len(X_initial)} 个样本")
print(f"最终数据集大小: {len(dante_optimizer.X_data)} 个样本")
print(f"新增样本数量: {len(dante_optimizer.X_data) - len(X_initial)}")
print(f"总迭代次数: {dante_optimizer.iteration_count}")
print(f"总科学计算次数: {sci_computer.calculation_count}")
print(f"模型版本数: {adaptive_model.model_version}")

# 性能改进
if 'initial_best' in locals() and 'final_best' in locals():
    performance_improvement = final_best['performance'] - initial_best['performance']
    relative_improvement = (performance_improvement / initial_best['performance']) * 100
    
    print(f"\n【性能改进】")
    print(f"初始最佳性能: {initial_best['performance']:.6f}")
    print(f"最终最佳性能: {final_best['performance']:.6f}")
    print(f"绝对改进: {performance_improvement:+.6f}")
    print(f"相对改进: {relative_improvement:+.2f}%")

# 成分变化分析
if 'initial_best' in locals() and 'final_best' in locals():
    print(f"\n【最佳成分变化】")
    elements = ['Co', 'Mo', 'Ti', 'Fe']
    for i, element in enumerate(elements):
        initial_val = initial_best['composition'][i]
        final_val = final_best['composition'][i]
        change = final_val - initial_val
        print(f"{element}: {initial_val:.2f}% → {final_val:.2f}% (变化: {change:+.2f}%)")

# 保存最终结果
final_results = {
    'optimization_summary': {
        'initial_dataset_size': len(X_initial),
        'final_dataset_size': len(dante_optimizer.X_data),
        'total_iterations': dante_optimizer.iteration_count,
        'total_calculations': sci_computer.calculation_count,
        'model_versions': adaptive_model.model_version
    },
    'performance_results': {
        'initial_best_performance': initial_best['performance'] if 'initial_best' in locals() else None,
        'final_best_performance': final_best['performance'],
        'improvement': performance_improvement if 'performance_improvement' in locals() else None,
        'relative_improvement_percent': relative_improvement if 'relative_improvement' in locals() else None
    },
    'best_composition': {
        'Co': final_best['composition'][0],
        'Mo': final_best['composition'][1], 
        'Ti': final_best['composition'][2],
        'Fe': final_best['composition'][3]
    },
    'optimization_history': dante_optimizer.optimization_history,
    'timestamp': datetime.now().isoformat()
}

# 保存结果到文件
with open('iterative_optimization_results.json', 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n【保存结果】")
print(f"详细结果已保存到: iterative_optimization_results.json")

# 保存最终模型
adaptive_model.save_model("final_adaptive_model")

print(f"\n" + "="*80)
print("DANTE迭代优化系统运行完成！")
print(f"="*80)
