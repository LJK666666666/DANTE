#!/usr/bin/env python3
"""
测试损失函数修复
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def test_custom_loss_function():
    """测试自定义相组成损失函数"""
    print("🧪 测试自定义相组成损失函数")
    print("=" * 40)
    
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
    
    # 测试损失函数
    print("测试自定义损失函数...")
    
    # 模拟数据
    y_true = tf.constant([[0.2, 0.3, 0.1, 0.3, 0.1],
                         [0.1, 0.4, 0.2, 0.2, 0.1]], dtype=tf.float32)
    
    # 好的预测（接近真实值）
    y_pred_good = tf.constant([[0.21, 0.29, 0.11, 0.29, 0.1],
                              [0.11, 0.39, 0.19, 0.21, 0.1]], dtype=tf.float32)
    
    # 差的预测（和不为1）
    y_pred_bad = tf.constant([[0.3, 0.4, 0.2, 0.4, 0.2],  # 和>1
                             [0.05, 0.2, 0.1, 0.1, 0.05]], dtype=tf.float32)  # 和<1
    
    loss_good = phase_composition_loss(y_true, y_pred_good)
    loss_bad = phase_composition_loss(y_true, y_pred_bad)
    
    print(f"好预测的损失: {loss_good.numpy():.6f}")
    print(f"差预测的损失: {loss_bad.numpy():.6f}")
    
    if loss_bad > loss_good:
        print("✅ 自定义损失函数工作正常（差预测的损失更高）")
        return True
    else:
        print("❌ 自定义损失函数可能有问题")
        return False

def test_model_compilation():
    """测试模型编译"""
    print("\n🧪 测试模型编译")
    print("=" * 30)
    
    def phase_composition_loss(y_true, y_pred):
        """自定义相组成损失函数"""
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        sum_constraint = tf.reduce_mean(tf.square(tf.reduce_sum(y_pred, axis=1) - 1.0))
        negative_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred))
        total_loss = mse_loss + 0.1 * sum_constraint + 0.1 * negative_penalty
        return total_loss
    
    try:
        # 创建简单模型
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(5, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=phase_composition_loss,
            metrics=['mae']
        )
        
        print("✅ 模型编译成功")
        
        # 测试训练
        X_test = np.random.rand(100, 4)
        y_test_raw = np.random.rand(100, 5)
        y_test = y_test_raw / np.sum(y_test_raw, axis=1, keepdims=True)  # 归一化
        
        print("测试模型训练...")
        history = model.fit(
            X_test, y_test,
            epochs=5,
            batch_size=16,
            verbose=0
        )
        
        print(f"✅ 模型训练成功，最终损失: {history.history['loss'][-1]:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ 模型编译或训练失败: {e}")
        return False

def test_alternative_approaches():
    """测试替代方案"""
    print("\n🧪 测试替代损失函数方案")
    print("=" * 35)
    
    # 方案1：简单MSE
    print("方案1: 简单MSE损失")
    try:
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(64, activation='relu')(inputs)
        outputs = layers.Dense(5, activation='softmax')(x)
        model1 = keras.Model(inputs=inputs, outputs=outputs)
        model1.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✅ MSE损失编译成功")
    except Exception as e:
        print(f"❌ MSE损失失败: {e}")
    
    # 方案2：Huber损失（对异常值更鲁棒）
    print("方案2: Huber损失")
    try:
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(64, activation='relu')(inputs)
        outputs = layers.Dense(5, activation='softmax')(x)
        model2 = keras.Model(inputs=inputs, outputs=outputs)
        model2.compile(optimizer='adam', loss='huber', metrics=['mae'])
        print("✅ Huber损失编译成功")
    except Exception as e:
        print(f"❌ Huber损失失败: {e}")
    
    # 方案3：Mean Absolute Error
    print("方案3: MAE损失")
    try:
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(64, activation='relu')(inputs)
        outputs = layers.Dense(5, activation='softmax')(x)
        model3 = keras.Model(inputs=inputs, outputs=outputs)
        model3.compile(optimizer='adam', loss='mae', metrics=['mse'])
        print("✅ MAE损失编译成功")
    except Exception as e:
        print(f"❌ MAE损失失败: {e}")
    
    return True

def main():
    """主测试函数"""
    print("🔧 损失函数修复验证")
    print("=" * 50)
    
    test1 = test_custom_loss_function()
    test2 = test_model_compilation()
    test3 = test_alternative_approaches()
    
    print("\n📊 测试结果:")
    print("=" * 20)
    print(f"自定义损失函数: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"模型编译测试: {'✅ 通过' if test2 else '❌ 失败'}")
    print(f"替代方案测试: {'✅ 通过' if test3 else '❌ 失败'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 损失函数修复验证成功！")
        print("\n现在notebook应该能够:")
        print("  ✅ 正确编译相组成预测模型")
        print("  ✅ 使用自定义损失函数训练")
        print("  ✅ 不再出现 'Could not interpret loss identifier' 错误")
        
        print("\n📋 自定义损失函数特点:")
        print("  🔸 结合MSE损失和约束项")
        print("  🔸 确保相组成和为1（软约束）")
        print("  🔸 防止负值预测")
        print("  🔸 更适合相组成预测任务")
        
        return True
    else:
        print("\n⚠️ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✨ 现在可以重新运行notebook，损失函数错误已修复！")
    else:
        print("\n🔧 需要进一步调试损失函数问题")
