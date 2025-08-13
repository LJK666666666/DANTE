#!/usr/bin/env python3
"""
测试双网络模型可视化代码
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def test_dual_network_visualization():
    """测试双网络模型可视化"""
    print("🎨 测试双网络模型可视化")
    print("=" * 40)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 621
    
    # 模拟真实值
    Y_elastic = np.random.uniform(8e10, 1.5e11, n_samples)  # 弹性模量范围
    Y_yield = np.random.uniform(5e8, 2e9, n_samples)        # 屈服强度范围
    
    # 模拟预测值（添加一些噪声但保持相关性）
    elastic_predictions = Y_elastic + np.random.normal(0, Y_elastic.std() * 0.1, n_samples)
    yield_predictions = Y_yield + np.random.normal(0, Y_yield.std() * 0.1, n_samples)
    
    print(f"生成模拟数据: {n_samples} 个样本")
    
    # 计算R²分数
    elastic_r2 = r2_score(Y_elastic, elastic_predictions)
    yield_r2 = r2_score(Y_yield, yield_predictions)
    
    elastic_mse = mean_squared_error(Y_elastic, elastic_predictions)
    yield_mse = mean_squared_error(Y_yield, yield_predictions)
    
    print(f"弹性模量预测 - R²: {elastic_r2:.4f}, MSE: {elastic_mse:.2e}")
    print(f"屈服强度预测 - R²: {yield_r2:.4f}, MSE: {yield_mse:.2e}")
    
    # 创建可视化图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 弹性模量预测效果
    ax1.scatter(Y_elastic, elastic_predictions, alpha=0.6, s=30, color='blue', edgecolors='navy', linewidth=0.5)
    
    # 添加理想预测线（y=x）
    min_elastic = min(Y_elastic.min(), elastic_predictions.min())
    max_elastic = max(Y_elastic.max(), elastic_predictions.max())
    ax1.plot([min_elastic, max_elastic], [min_elastic, max_elastic], 'r--', linewidth=2, label='理想预测线 (y=x)')
    
    ax1.set_xlabel('实际弹性模量 (Pa)', fontsize=12)
    ax1.set_ylabel('预测弹性模量 (Pa)', fontsize=12)
    ax1.set_title(f'弹性模量网络预测效果\nR² = {elastic_r2:.4f}, MSE = {elastic_mse:.2e}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 设置科学计数法
    ax1.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # 屈服强度预测效果
    ax2.scatter(Y_yield, yield_predictions, alpha=0.6, s=30, color='green', edgecolors='darkgreen', linewidth=0.5)
    
    # 添加理想预测线（y=x）
    min_yield = min(Y_yield.min(), yield_predictions.min())
    max_yield = max(Y_yield.max(), yield_predictions.max())
    ax2.plot([min_yield, max_yield], [min_yield, max_yield], 'r--', linewidth=2, label='理想预测线 (y=x)')
    
    ax2.set_xlabel('实际屈服强度 (Pa)', fontsize=12)
    ax2.set_ylabel('预测屈服强度 (Pa)', fontsize=12)
    ax2.set_title(f'屈服强度网络预测效果\nR² = {yield_r2:.4f}, MSE = {yield_mse:.2e}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 设置科学计数法
    ax2.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 保存图表
    try:
        fig.savefig('test_dual_network_prediction_performance.png', dpi=300, bbox_inches='tight')
        print("✅ 测试图表已保存为 'test_dual_network_prediction_performance.png'")
    except Exception as e:
        print(f"⚠️ 保存图表失败: {e}")
    
    plt.show()
    
    # 显示预测统计信息
    print(f"\n=== 双网络模型预测统计 ===")
    print(f"数据点总数: {len(Y_elastic)}")
    print(f"\n弹性模量网络:")
    print(f"  实际值范围: {Y_elastic.min():.2e} - {Y_elastic.max():.2e} Pa")
    print(f"  预测值范围: {elastic_predictions.min():.2e} - {elastic_predictions.max():.2e} Pa")
    print(f"  平均绝对误差: {np.mean(np.abs(Y_elastic - elastic_predictions)):.2e} Pa")
    print(f"  相对误差 (MAPE): {np.mean(np.abs((Y_elastic - elastic_predictions) / Y_elastic)) * 100:.2f}%")
    
    print(f"\n屈服强度网络:")
    print(f"  实际值范围: {Y_yield.min():.2e} - {Y_yield.max():.2e} Pa")
    print(f"  预测值范围: {yield_predictions.min():.2e} - {yield_predictions.max():.2e} Pa")
    print(f"  平均绝对误差: {np.mean(np.abs(Y_yield - yield_predictions)):.2e} Pa")
    print(f"  相对误差 (MAPE): {np.mean(np.abs((Y_yield - yield_predictions) / Y_yield)) * 100:.2f}%")
    
    return True

def test_visualization_with_real_data():
    """使用真实数据测试可视化"""
    print("\n🎨 使用真实数据测试可视化")
    print("=" * 40)
    
    try:
        # 加载真实数据
        data = pd.read_csv('data.csv')
        print(f"✅ 加载真实数据: {len(data)} 个样本")
        
        # 提取真实的弹性模量和屈服强度
        Y_elastic_real = data['elastic'].values
        Y_yield_real = data['yield'].values
        
        # 模拟预测值（在真实数据基础上添加噪声）
        elastic_pred_real = Y_elastic_real + np.random.normal(0, Y_elastic_real.std() * 0.15, len(Y_elastic_real))
        yield_pred_real = Y_yield_real + np.random.normal(0, Y_yield_real.std() * 0.15, len(Y_yield_real))
        
        # 计算性能指标
        elastic_r2_real = r2_score(Y_elastic_real, elastic_pred_real)
        yield_r2_real = r2_score(Y_yield_real, yield_pred_real)
        
        print(f"真实数据预测性能:")
        print(f"  弹性模量 R²: {elastic_r2_real:.4f}")
        print(f"  屈服强度 R²: {yield_r2_real:.4f}")
        
        # 创建真实数据的可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 弹性模量
        ax1.scatter(Y_elastic_real, elastic_pred_real, alpha=0.6, s=30, color='blue', edgecolors='navy', linewidth=0.5)
        min_e = min(Y_elastic_real.min(), elastic_pred_real.min())
        max_e = max(Y_elastic_real.max(), elastic_pred_real.max())
        ax1.plot([min_e, max_e], [min_e, max_e], 'r--', linewidth=2, label='理想预测线')
        ax1.set_xlabel('实际弹性模量 (Pa)')
        ax1.set_ylabel('预测弹性模量 (Pa)')
        ax1.set_title(f'真实数据 - 弹性模量预测\nR² = {elastic_r2_real:.4f}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        # 屈服强度
        ax2.scatter(Y_yield_real, yield_pred_real, alpha=0.6, s=30, color='green', edgecolors='darkgreen', linewidth=0.5)
        min_y = min(Y_yield_real.min(), yield_pred_real.min())
        max_y = max(Y_yield_real.max(), yield_pred_real.max())
        ax2.plot([min_y, max_y], [min_y, max_y], 'r--', linewidth=2, label='理想预测线')
        ax2.set_xlabel('实际屈服强度 (Pa)')
        ax2.set_ylabel('预测屈服强度 (Pa)')
        ax2.set_title(f'真实数据 - 屈服强度预测\nR² = {yield_r2_real:.4f}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        plt.tight_layout()
        fig.savefig('test_real_data_prediction_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 真实数据可视化测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 双网络模型可视化测试")
    print("=" * 50)
    
    # 测试1：模拟数据
    success1 = test_dual_network_visualization()
    
    # 测试2：真实数据
    success2 = test_visualization_with_real_data()
    
    if success1 and success2:
        print("\n🎉 所有可视化测试通过！")
        print("✅ 可视化代码已准备就绪，可以在notebook中使用")
    else:
        print("\n⚠️ 部分测试失败，但基本功能正常")
