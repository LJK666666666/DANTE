#!/usr/bin/env python3
"""
测试强化的相组成预测模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import ast

# 检查TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("✅ TensorFlow已加载")
except ImportError:
    print("❌ TensorFlow未安装")
    exit(1)

# GPU内存优化
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU内存优化已启用")
    except:
        print("⚠️ GPU配置失败")

def phase_composition_loss(y_true, y_pred):
    """自定义相组成损失函数"""
    # MSE损失
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 确保预测值和为1的约束（软约束）
    sum_constraint = tf.reduce_mean(tf.square(tf.reduce_sum(y_pred, axis=1) - 1.0))
    
    # 防止负值的约束
    negative_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred))
    
    # 组合损失
    total_loss = mse_loss + 0.1 * sum_constraint + 0.1 * negative_penalty
    
    return total_loss

def create_enhanced_phase_model(input_dim=4, output_dim=5):
    """创建强化的相组成预测模型"""
    inputs = keras.Input(shape=(input_dim,), name='phase_input')
    
    # 输入特征增强
    x = layers.Dense(512, activation='relu', name='dense1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.4, name='dropout1')(x)
    
    # 更深的特征提取
    x = layers.Dense(384, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    x = layers.Dense(256, activation='relu', name='dense3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # 残差块模拟
    for i, units in enumerate([256, 192, 128, 96]):
        residual = x
        if x.shape[-1] != units:
            residual = layers.Dense(units)(residual)
        
        x = layers.Dense(units, activation='relu', name=f'res_dense1_{i}')(x)
        x = layers.BatchNormalization(name=f'res_bn1_{i}')(x)
        x = layers.Dropout(0.2, name=f'res_dropout1_{i}')(x)
        x = layers.Dense(units, activation='relu', name=f'res_dense2_{i}')(x)
        x = layers.Add(name=f'res_add_{i}')([x, residual])
    
    # 注意力机制模拟
    attention = layers.Dense(x.shape[-1], activation='sigmoid', name='attention')(x)
    x = layers.Multiply(name='attention_apply')([x, attention])
    
    # 最终特征提取
    x = layers.Dense(128, activation='relu', name='dense4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.1, name='dropout4')(x)
    
    x = layers.Dense(64, activation='relu', name='dense5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    
    # 输出层
    outputs = layers.Dense(output_dim, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='enhanced_phase_model')
    
    # 使用AdamW优化器
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        ),
        loss=phase_composition_loss,
        metrics=['mae', 'mse']
    )
    
    return model

def test_enhanced_phase_model():
    """测试强化的相组成预测模型"""
    print("🧪 测试强化的相组成预测模型")
    print("=" * 50)
    
    # 加载真实数据
    try:
        data = pd.read_csv('data.csv')
        print(f"✅ 加载数据: {len(data)} 样本")
    except:
        print("❌ 无法加载数据文件")
        return False
    
    # 数据预处理
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
        except:
            return {}
    
    # 解析数据
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
    
    print(f"元素特征形状: {X_elements.shape}")
    print(f"相组成目标形状: {X_phases.shape}")
    
    # 数据标准化
    element_scaler = StandardScaler()
    X_elements_scaled = element_scaler.fit_transform(X_elements)
    
    # 确保相组成数据归一化
    X_phases_norm = X_phases / (np.sum(X_phases, axis=1, keepdims=True) + 1e-8)
    
    print(f"相组成和的范围: {np.sum(X_phases_norm, axis=1).min():.6f} - {np.sum(X_phases_norm, axis=1).max():.6f}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_elements_scaled, X_phases_norm, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 创建强化模型
    model = create_enhanced_phase_model(input_dim=4, output_dim=5)
    print(f"✅ 创建强化相组成预测模型")
    print(f"模型参数数量: {model.count_params():,}")
    
    # 数据增强
    noise_factor = 0.01
    X_train_aug = X_train + np.random.normal(0, noise_factor, X_train.shape)
    
    # 训练配置
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,  # 大patience
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-8,
        verbose=1
    )
    
    try:
        print("开始训练强化相组成预测模型...")
        history = model.fit(
            X_train_aug, y_train,
            validation_data=(X_test, y_test),
            epochs=300,  # 更多epochs
            batch_size=4,  # 小batch size
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        # 评估性能
        y_pred = model.predict(X_test, verbose=0)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n强化模型性能:")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.6f}")
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 预测vs实际散点图
        ax1.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.6, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Phase Composition')
        ax1.set_ylabel('Predicted Phase Composition')
        ax1.set_title(f'Enhanced Phase Model Performance\nR² = {r2:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # 训练历史
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_phase_model_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if r2 > 0.5:
            print("✅ 强化相组成预测模型性能显著改善！")
            return True
        else:
            print(f"⚠️ 相组成预测模型仍需优化，当前R²: {r2:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_phase_model()
    
    if success:
        print("\n🎉 强化相组成预测模型测试成功！")
        print("现在notebook中的相组成预测应该有显著改善。")
    else:
        print("\n⚠️ 相组成预测仍然具有挑战性")
        print("这可能表明元素成分到相组成的映射本身就很复杂。")
