#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雷达图绘制脚本 - 展示训练前后的指标对比（优化美化版）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """创建雷达图"""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
    
    register_projection(RadarAxes)
    return theta


def plot_radar_chart(metrics, before_values, after_values, save_path='radar_chart_enhanced.png'):
    """绘制优化美化后的雷达图"""
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')  # 使用更现代的样式
    
    # 创建雷达图
    num_vars = len(metrics)
    theta = radar_factory(num_vars, frame='polygon')
    
    # 创建图形（更大的画布，白色背景）
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='radar', facecolor='#f8f9fa')
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92)
    
    # 使用更现代的配色方案
    color_before = '#3498db'  # 蓝色
    color_after = '#e74c3c'   # 红色
    
    # 绘制训练前的数据（带阴影效果）
    ax.plot(theta, before_values, 'o-', linewidth=3, 
            color=color_before, label='训练前', 
            markersize=10, markeredgewidth=2, markeredgecolor='white',
            zorder=3)
    ax.fill(theta, before_values, alpha=0.25, color=color_before)
    
    # 绘制训练后的数据（带阴影效果）
    ax.plot(theta, after_values, 's-', linewidth=3, 
            color=color_after, label='训练后',
            markersize=10, markeredgewidth=2, markeredgecolor='white',
            zorder=3)
    ax.fill(theta, after_values, alpha=0.25, color=color_after)
    
    # 在每个数据点上添加数值标注
    for i, (angle, before_val, after_val) in enumerate(zip(theta, before_values, after_values)):
        # 训练前数值
        ax.text(angle, before_val + 3, f'{before_val:.1f}', 
                ha='center', va='bottom', fontsize=10, 
                color=color_before, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_before, alpha=0.8))
        
        # 训练后数值
        ax.text(angle, after_val + 3, f'{after_val:.1f}', 
                ha='center', va='bottom', fontsize=10, 
                color=color_after, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_after, alpha=0.8))
    
    # 设置标签（更大的字体）
    ax.set_varlabels(metrics)
    for label in ax.get_xticklabels():
        label.set_fontsize(13)
        label.set_fontweight('bold')
    
    # 设置y轴范围和刻度
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=11, color='gray')
    
    # 美化网格
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.4, color='gray')
    ax.spines['polar'].set_color('#cccccc')
    ax.spines['polar'].set_linewidth(2)
    
    # 添加图例（更大更美观）
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), 
                      fontsize=14, frameon=True, shadow=True, 
                      fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # 添加标题（更大更醒目）
    plt.title('训练前后指标对比雷达图', 
             size=20, weight='bold', pad=30, 
             color='#2c3e50',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', 
                      edgecolor='#95a5a6', linewidth=2))
    
    # 添加副标题（提升信息）
    improvements = [after - before for before, after in zip(before_values, after_values)]
    avg_improvement = np.mean(improvements)
    plt.text(0.5, 0.02, f'平均提升: {avg_improvement:.2f}%', 
            ha='center', va='bottom', transform=fig.transFigure,
            fontsize=12, color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f4e6', 
                     edgecolor='#27ae60', linewidth=1.5))
    
    # 保存图片（高分辨率）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 优化后的雷达图已保存到: {save_path}")
    
    # 显示图片
    plt.show()


if __name__ == "__main__":
    # 示例数据：指标名称
    metrics = [
        'Tool Acc',
        'Schema Valid',
        'Query Param Acc',
        'Obs Param Acc',
        'Duplicate Rate',
        'State Consistency'
    ]
    
    # 示例数据：训练前的数值（百分比，0-100）
    before_values = [65.5, 72.3, 68.9, 61.2, 85.4, 70.8]
    
    # 示例数据：训练后的数值（百分比，0-100）
    after_values = [82.3, 88.7, 85.1, 79.6, 92.3, 86.5]
    
    # 打印数据表格
    print("="*60)
    print("训练前后指标对比表")
    print("="*60)
    print(f"{'指标':<20} {'训练前':<10} {'训练后':<10} {'提升':<10}")
    print("-"*60)
    for i, metric in enumerate(metrics):
        improvement = after_values[i] - before_values[i]
        print(f"{metric:<20} {before_values[i]:<10.2f} {after_values[i]:<10.2f} {improvement:<10.2f}")
    print("="*60)
    
    # 绘制优化后的雷达图
    plot_radar_chart(metrics, before_values, after_values, 'radar_chart_enhanced.png')
