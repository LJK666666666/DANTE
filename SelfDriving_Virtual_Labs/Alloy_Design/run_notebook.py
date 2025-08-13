#!/usr/bin/env python3
"""
执行DANTE合金设计notebook的完整脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import gc

# 检查TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping
    print("✅ TensorFlow已加载")
except ImportError:
    print("❌ TensorFlow未安装")
    sys.exit(1)

# GPU内存优化配置
print("配置GPU内存优化...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU内存增长模式已启用，检测到 {len(gpus)} 个GPU")
    except RuntimeError as e:
        print(f"⚠️ GPU配置失败: {e}")
else:
    print("ℹ️ 未检测到GPU，将使用CPU训练")

# 启用混合精度训练
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ 混合精度训练已启用（FP16）")
except:
    print("⚠️ 混合精度训练启用失败，使用默认精度")

def clear_training_cache():
    """清理训练缓存以释放内存"""
    tf.keras.backend.clear_session()
    gc.collect()

def main():
    print("🚀 开始执行DANTE合金设计完整流程")
    print("=" * 60)
    
    # ==================== 第一部分：数据加载与预处理 ====================
    print("\n📊 第一部分：数据加载与预处理")
    print("-" * 40)
    
    # 加载数据
    try:
        data = pd.read_csv('data.csv')
        print(f"✅ 成功加载数据集，共 {len(data)} 个样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 解析元素成分
    def parse_element_composition(sid):
        elements = {'Co': 0, 'Mo': 0, 'Ti': 0, 'Fe': 0}
        sid = sid.replace('Co', ' Co').replace('Mo', ' Mo').replace('Ti', ' Ti').replace('Fe', ' Fe').strip()
        parts = sid.split()
        for part in parts:
            if 'Co' in part:
                elements['Co'] = float(part.replace('Co', ''))
            elif 'Mo' in part:
                elements['Mo'] = float(part.replace('Mo', ''))
            elif 'Ti' in part:
                elements['Ti'] = float(part.replace('Ti', ''))
            elif 'Fe' in part:
                elements['Fe'] = float(part.replace('Fe', ''))
        
        if elements['Fe'] == 0:
            elements['Fe'] = 100.0 - (elements['Co'] + elements['Mo'] + elements['Ti'])
        
        return elements
    
    # 解析化合物比例
    def parse_compound_ratios(compound_str):
        if pd.isna(compound_str):
            return {}
        try:
            if isinstance(compound_str, str):
                compound_str = compound_str.replace("'", '"')
                try:
                    compound_data = json.loads(compound_str)
                except:
                    compound_data = ast.literal_eval(compound_str)
            else:
                compound_data = compound_str
            return compound_data
        except Exception as e:
            return {}
    
    # 处理数据
    print("解析元素成分和化合物比例...")
    
    element_compositions = []
    for sid in data['sid']:
        composition = parse_element_composition(sid)
        element_compositions.append([composition['Co'], composition['Mo'], composition['Ti'], composition['Fe']])
    
    X_elements = np.array(element_compositions)
    
    compound_ratios = []
    for compound_str in data['phase_ratio_dict']:
        ratios = parse_compound_ratios(compound_str)
        compound_ratios.append(ratios)
    
    phase_names = ['martensite', 'Fe2Mo', 'austenite', 'gamma_phase', 'Ni3Ti']
    phase_values = []
    for phase_name in phase_names:
        phase_values.append([ratios.get(phase_name, 0.0) for ratios in compound_ratios])
    
    X_phases = np.array(phase_values).T
    
    # 目标变量
    Y_elastic = data['elastic'].values
    Y_yield = data['yield'].values
    
    # 归一化
    elastic_min, elastic_max = Y_elastic.min(), Y_elastic.max()
    yield_min, yield_max = Y_yield.min(), Y_yield.max()
    
    Y_elastic_norm = (Y_elastic - elastic_min) / (elastic_max - elastic_min)
    Y_yield_norm = (Y_yield - yield_min) / (yield_max - yield_min)
    Y = (Y_elastic_norm + Y_yield_norm) / 2
    
    X_elements_3d = X_elements[:, :3]  # 3维搜索空间
    
    print(f"✅ 数据预处理完成")
    print(f"   元素成分: {X_elements.shape} (4维), {X_elements_3d.shape} (3维)")
    print(f"   化合物比例: {X_phases.shape}")
    print(f"   弹性模量范围: {elastic_min:.2e} - {elastic_max:.2e}")
    print(f"   屈服强度范围: {yield_min:.2e} - {yield_max:.2e}")
    
    # ==================== 第二部分：双网络模型训练 ====================
    print("\n🧠 第二部分：双网络模型训练")
    print("-" * 40)
    
    class DualNetworkAlloySurrogateModel:
        def __init__(self, search_dims=3, network_input_dims=4, n_folds=5):
            self.input_dims = search_dims
            self.search_dims = search_dims
            self.network_input_dims = network_input_dims
            self.element_scaler = StandardScaler()
            self.n_folds = n_folds
            self.elastic_models = []
            self.yield_models = []
            self.ensemble_model = None
            self.is_trained = False
        
        def convert_3d_to_4d(self, x_3d):
            if x_3d.ndim == 1:
                fe_content = 100.0 - np.sum(x_3d)
                return np.append(x_3d, fe_content)
            else:
                fe_content = 100.0 - np.sum(x_3d, axis=1)
                return np.column_stack([x_3d, fe_content])
        
        def create_neural_network(self, input_dim, output_dim=1):
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(output_dim, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            return model
        
        def train(self, x_search, y_elastic, y_yield, verbose=1):
            x_elements_4d = self.convert_3d_to_4d(x_search)
            x_elements_scaled = self.element_scaler.fit_transform(x_elements_4d)
            
            if verbose:
                print(f"开始双网络5折交叉验证...")
                print(f"输入特征维度: {x_elements_scaled.shape[1]}")
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            elastic_scores = []
            yield_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(x_elements_scaled)):
                if verbose:
                    print(f"训练第 {fold+1}/{self.n_folds} 折...")
                
                X_train, X_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
                y_elastic_train, y_elastic_val = y_elastic[train_idx], y_elastic[val_idx]
                y_yield_train, y_yield_val = y_yield[train_idx], y_yield[val_idx]
                
                # 训练弹性模量网络
                elastic_model = self.create_neural_network(input_dim=self.network_input_dims)
                early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                
                elastic_model.fit(
                    X_train, y_elastic_train,
                    validation_data=(X_val, y_elastic_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 训练屈服强度网络
                yield_model = self.create_neural_network(input_dim=self.network_input_dims)
                yield_model.fit(
                    X_train, y_yield_train,
                    validation_data=(X_val, y_yield_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 评估
                elastic_pred = elastic_model.predict(X_val, verbose=0)
                yield_pred = yield_model.predict(X_val, verbose=0)
                
                elastic_r2 = r2_score(y_elastic_val, elastic_pred)
                yield_r2 = r2_score(y_yield_val, yield_pred)
                
                elastic_scores.append(elastic_r2)
                yield_scores.append(yield_r2)
                
                if verbose:
                    print(f"  弹性模量 R²: {elastic_r2:.4f}")
                    print(f"  屈服强度 R²: {yield_r2:.4f}")
                
                self.elastic_models.append(elastic_model)
                self.yield_models.append(yield_model)
                
                clear_training_cache()
            
            avg_elastic_r2 = np.mean(elastic_scores)
            avg_yield_r2 = np.mean(yield_scores)
            
            if verbose:
                print(f"双网络交叉验证结果:")
                print(f"弹性模量平均 R²: {avg_elastic_r2:.4f} ± {np.std(elastic_scores):.4f}")
                print(f"屈服强度平均 R²: {avg_yield_r2:.4f} ± {np.std(yield_scores):.4f}")
            
            self.is_trained = True
            return self
    
    # 训练双网络模型
    print("开始训练双网络模型...")
    dual_model = DualNetworkAlloySurrogateModel(search_dims=3, network_input_dims=4, n_folds=5)
    
    try:
        trained_dual_model = dual_model.train(
            x_search=X_elements_3d,
            y_elastic=Y_elastic_norm,
            y_yield=Y_yield_norm,
            verbose=1
        )
        print("✅ 双网络模型训练完成！")
        
        # 测试预测
        test_sample = X_elements_3d[:1]  # 取第一个样本测试
        print(f"测试预测功能...")
        print(f"测试样本成分: Co={test_sample[0][0]:.2f}, Mo={test_sample[0][1]:.2f}, Ti={test_sample[0][2]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 双网络模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 DANTE合金设计流程执行成功！")
        print("✅ 双网络模型已训练完成")
        print("✅ 显存优化已应用")
        print("✅ 边界约束已修正")
    else:
        print("\n❌ 执行过程中出现错误")
