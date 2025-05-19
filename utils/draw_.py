import numpy as np
import matplotlib.pyplot as plt
import torch
def darw_(ground_truth:torch.Tensor, pred:torch.Tensor):
    # 确保两个张量的形状相同
    assert ground_truth.shape == pred.shape

    # 将GPU张量移动到CPU并转换为NumPy数组
    ground_truth_cpu = ground_truth.cpu().numpy()
    pred_cpu = pred.cpu().numpy()

    # 绘制两个张量的数据
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truth_cpu, label='Ground Truth')
    plt.plot(pred_cpu, label='Prediction', linestyle='--')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Ground Truth vs Prediction')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')

    # 显示图表
    plt.savefig('ground_truth_vs_pred.png')


import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_metrics(csv_files):
    """
    绘制多个CSV文件中训练指标随epoch变化的曲线

    参数：
    csv_files : list of str
        包含需要读取的CSV文件路径的列表
    """
    plt.figure(figsize=(12, 8))

    # 创建三个子图
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    # 颜色循环
    colors = plt.cm.tab10.colors

    for idx, file_path in enumerate(csv_files):
        try:
            # 读取数据时跳过首行参数说明
            df = pd.read_csv(file_path, skiprows=1)  # 关键修改点

            # 提取文件名作为标签
            label = os.path.splitext(os.path.basename(file_path))[0]

            # 绘制三个指标
            ax1.plot(df['epoch'], df['train_loss_iterated'],
                     color=colors[idx % 10], label=label, alpha=0.8)
            ax2.plot(df['epoch'], df['train_mse'],
                     color=colors[idx % 10], alpha=0.8)
            ax3.plot(df['epoch'], df['train_nse'],
                     color=colors[idx % 10], alpha=0.8)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    # 设置公共参数
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    # 设置子图专属参数
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Metrics Evolution')
    ax2.set_ylabel('MSE')
    ax3.set_ylabel('NSE')

    # 将图例放在所有子图之外
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels,
                  loc='upper center',
                  ncol=3,
                  bbox_to_anchor=(0.5, 1.02),
                  frameon=False)

    plt.tight_layout()
    plt.show()


def inference_plot(rain_fall, pred: torch.Tensor, ground_truth: torch.Tensor, iter_, idx = 0,figures_folder = 'figures_jhhhhc_r10',time_step = '10 min',save = False):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    n_step_0 = len(rain_fall[idx])
    x_0 = np.arange(n_step_0)
    n_step_1 = len(pred[idx])
    x_1 = np.arange(n_step_1)
    # 确保两个张量的形状相同
    assert ground_truth.shape == pred.shape

    # 将GPU张量移动到CPU并转换为NumPy数组
    ground_truth_cpu = ground_truth.cpu().numpy()
    pred_cpu = pred.cpu().numpy()
    rain_fall = rain_fall.cpu().numpy()
    # 定义保存图像的文件夹名称
    # figures_folder = 'figures___'

    # 如果文件夹不存在，则创建它
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # 基础文件名
    base_filename = f'ground_truth_vs_pred{iter_}.png'

    # 初始化后缀
    suffix = 1

    # 检查文件是否存在，如果存在则添加后缀
    filename = os.path.join(figures_folder, base_filename)
    while os.path.exists(filename):
        filename = os.path.join(figures_folder, f"{os.path.splitext(base_filename)[0]}_{suffix}{os.path.splitext(base_filename)[1]}")
        suffix += 1

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制从上至下的条形图
    ax1.bar(x_0, rain_fall[idx, :, 0], color='blue', label='rain_fall')
    ax1.set_ylabel('降雨强度(mm/h)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper right')

    # 调整y轴的刻度方向为从上至下
    ax1.set_ylim(ax1.get_ylim()[::-1])

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制从下至上的曲线图
    ax2.plot(x_1, ground_truth_cpu[idx, :, 0], color='red', marker='o', label='Ground Truth')
    ax2.set_ylabel('排口流量(CMS)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    # 绘制从下至上的第二条曲线图
    ax2.plot(x_1, pred_cpu[idx, :, 0], color='green', marker='x', label='Prediction')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.8))  # 更新图例

    # 设置x轴标签
    ax1.set_xlabel(f'时间步（间隔{time_step}）')

    # 调整布局
    plt.tight_layout()
    plt.show()
    # 保存图表
    if save:
        plt.savefig(filename)
        print(f"Image saved as {filename}")
    plt.close()  # 关闭图表以避免重叠


import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_single_inference_pre(ax, rain_fall_np, pred_np, gt_np, idx, time_step):
    """在给定的axes上绘制单个推理结果"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_facecolor('white')  # 子图背景设为白色
    # 计算时间步长
    n_step_0 = rain_fall_np[idx].shape[0]
    x_0 = np.arange(n_step_0)
    n_step_1 = pred_np[idx].shape[0]
    x_1 = np.arange(n_step_1)

    # 绘制降雨条形图（主坐标轴）
    ax1 = ax
    bars = ax1.bar(x_0, rain_fall_np[idx, :, 0], color='skyblue', alpha=0.7, label='降雨强度')
    ax1.set_ylabel('降雨强度 (mm/h)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(ax1.get_ylim()[::-1])  # 翻转Y轴

    # 创建副坐标轴（共享X轴）
    ax2 = ax1.twinx()

    # 绘制地面真实值和预测值
    line1 = ax2.plot(x_1, gt_np[idx, :, 0], 'r-', marker='o', markersize=4, label='真实值')
    line2 = ax2.plot(x_1, pred_np[idx, :, 0], 'g--', marker='x', markersize=4, label='预测值')
    ax2.set_ylabel('排口流量 (CMS)', color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.grid(linestyle='--', alpha=0.5)

    # 合并图例
    lines = line1 + line2 + [bars]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right',
               bbox_to_anchor=(1, 1), framealpha=0.9)

    # 设置公共参数
    ax.set_xlabel(f'时间步（间隔{time_step}）')
    plt.grid(False)
    # ax.set_title(f'Sample #{idx + 1}')

def plot_single_inference_td(ax, rain_fall_np, pred_np, gt_np, idx, time_step):
    """在给定的axes上绘制单个推理结果（排除填充步长）"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_facecolor('white')  # 子图背景设为白色
    # 获取当前样本数据
    rain_sample = rain_fall_np[idx, :, 0]  # 取第一个特征
    pred_sample = pred_np[idx, :, 0]
    gt_sample = gt_np[idx, :, 0]

    # 创建有效步长掩码（假设填充值为1）
    valid_mask = (rain_sample != 1)

    # 应用掩码过滤填充步长
    rain_filtered = rain_sample[valid_mask]
    pred_filtered = pred_sample[valid_mask]
    gt_filtered = gt_sample[valid_mask]

    # 生成有效时间步坐标
    n_steps = len(rain_filtered)
    x = np.arange(n_steps)

    # 绘制降雨条形图（主坐标轴）
    ax1 = ax
    bars = ax1.bar(x, rain_filtered, color='skyblue', alpha=0.7, label='降雨强度')
    ax1.set_ylabel('降雨强度 (mm/h)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(ax1.get_ylim()[::-1])  # 翻转Y轴

    # 创建副坐标轴（共享X轴）
    ax2 = ax1.twinx()
    # 设置坐标轴边框
    ax2.spines['top'].set_visible(True)  # 顶部边框可见
    ax2.spines['bottom'].set_visible(True)  # 底部边框可见
    ax2.spines['left'].set_visible(True)  # 左侧边框可见
    ax2.spines['right'].set_visible(True)  # 右侧边框可见

    # 可选：调整边框的颜色和宽度
    ax2.spines['top'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')

    ax2.spines['top'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['left'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)
    # 绘制地面真实值（灰色圆圈散点）和预测值（保持绿色虚线）
    line1 = ax2.plot(x, gt_filtered, 'o', color='gray', markersize=4, linestyle='', label='真实值')  # 修改处
    line2 = ax2.plot(x, pred_filtered, 'g-', marker='x', markersize=4, label='预测值')
    ax2.set_ylabel('排口流量 (CMS)', color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.grid(linestyle='-', alpha=0.5)

    # 合并图例
    lines = line1 + line2 + [bars]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right',
               bbox_to_anchor=(1, 1), framealpha=0.9)

    # 设置公共参数
    ax.set_xlabel(f'时间步（间隔{time_step}）')
    plt.grid(False)

def plot_multiple_inference(rain_fall, pred, ground_truth, iter_, idx_list=None,
                            figures_folder='figures_jhhhhc_r10', time_step='10 min',
                            save=False):
    """绘制6个子图的综合可视化"""
    # 转换为numpy数组
    rain_fall_np = rain_fall.cpu().numpy()
    pred_np = pred.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    # 创建画布和子图
    fig, axs = plt.subplots(3, 2, figsize=(18, 15))
    # fig.suptitle(f'Model Predictions vs Ground Truth (Iteration {iter_})',
    #              fontsize=16, y=1.02)

    # 默认绘制前6个样本
    if idx_list is None:
        idx_list = list(range(6))
    else:
        assert len(idx_list) == 6, "需要6个索引值"

    # 遍历每个子图进行绘制
    for i, ax in enumerate(axs.flat):
        if rain_fall_np.shape[1] == pred_np.shape[1]:
            plot_single_inference_td(ax, rain_fall_np, pred_np, gt_np, idx_list[i], time_step)
        else:
            plot_single_inference_pre(ax, rain_fall_np, pred_np, gt_np, idx_list[i], time_step)
        ax.set_xlim(left=0)  # 统一X轴起点

    # 调整布局
    plt.tight_layout()

    # 保存结果
    if save:
        os.makedirs(figures_folder, exist_ok=True)
        filename = os.path.join(figures_folder, f'combined_plot_{iter_}.svg')
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to {filename}')
    plt.grid(False)
    plt.show()
    plt.close()



# 使用示例
# 假设已有以下数据：
# rain_fall = torch.randn(10, 24, 1)  # 示例数据（10个样本，24个时间步）
# pred = torch.randn(10, 12, 1)      # 预测值（12个预测步）
# ground_truth = torch.randn(10, 12, 1)

# 调用函数绘制
# plot_multiple_inference(rain_fall, pred, ground_truth, iter_=100, save=True)

def plot_single_inference_lft(ax, rain_fall_np, pred_np, pred_np1, pred_np2, pred_np3, pred_np4, pred_np5, gt_np, idx,
                              time_step):
    """在给定的axes上绘制单个推理结果（排除填充步长）"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_facecolor('white')  # 子图背景设为白色

    # 获取当前样本数据
    rain_sample = rain_fall_np[idx, :, 0]  # 取第一个特征
    pred_sample = pred_np[idx, :, 0]
    gt_sample = gt_np[idx, :, 0]

    # 创建有效步长掩码（假设填充值为1）
    valid_mask = (rain_sample != 1)

    # 应用掩码过滤填充步长
    rain_filtered = rain_sample[valid_mask]
    pred_filtered = pred_sample[valid_mask]
    gt_filtered = gt_sample[valid_mask]

    # 生成有效时间步坐标
    n_steps = len(rain_filtered)
    x = np.arange(n_steps)
    # ---------------------- 新增刻度线设置 ----------------------
    # 配置主刻度参数
    ax.tick_params(
        axis='x',  # 选择x轴
        which='both',  # 同时影响主次刻度（这里主刻度需要显示）
        direction='out',  # 刻度线朝外
        length=6,  # 主刻度线长度
        width=1,  # 刻度线宽度
        color='black',  # 刻度线颜色
        bottom=True,  # 显示底部刻度线
        top=False  # 不显示顶部刻度线
    )
    # 设置x轴刻度（核心新增代码）---------------------------------
    from matplotlib.ticker import MaxNLocator

    # 方式1：自动调整刻度数量（约6-8个刻度）
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    # 绘制降雨条形图（主坐标轴）
    ax1 = ax
    bars = ax1.bar(x, rain_filtered, color='skyblue', alpha=0.7, label='降雨强度')
    ax1.set_ylabel('降雨强度 (mm/h)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(ax1.get_ylim()[::-1])  # 翻转Y轴

    # 创建副坐标轴（共享X轴）
    ax2 = ax1.twinx()

    # 绘制地面真实值（灰色圆圈散点）
    line1 = ax2.plot(x, gt_filtered, 'o', color='gray', markersize=4, linestyle='', label='真实值')

    # 绘制主预测值（绿色虚线）
    # line2 = ax2.plot(x, pred_filtered, 'g-', marker='x', markersize=4, label='Lossfun1')
    line2 = ax2.plot(x, pred_filtered, color = '#008080', linestyle = '--', marker='x', markersize=4, label='训练方法1')

    # 收集并绘制额外预测值（pred_np1到pred_np5）
    additional_preds = [pred_np1, pred_np2, pred_np3, pred_np4, pred_np5]
    additional_preds = [p for p in additional_preds if p is not None]  # 过滤空值

    # 颜色和线型配置
    colors = ['blue', 'red', 'purple', 'orange', 'cyan']  # 为5条线预定义颜色
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]  # 不同线型

    additional_lines = []
    for i, pred in enumerate(additional_preds):
        # 获取预测数据
        pred_sampleX = pred[idx, :, 0]
        pred_filteredX = pred_sampleX[valid_mask]

        # 绘制带不同样式的新线
        new_line = ax2.plot(
            x, pred_filteredX,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker='.',
            markersize=4,
            label=f'训练方法{i + 2}'
        )
        additional_lines.append(new_line[0])

    # 设置副坐标轴参数
    ax2.set_ylabel('排口流量 (CMS)', color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.grid(linestyle='-', alpha=0.5)
    # 设置坐标轴边框
    ax2.spines['top'].set_visible(True)  # 顶部边框可见
    ax2.spines['bottom'].set_visible(True)  # 底部边框可见
    ax2.spines['left'].set_visible(True)  # 左侧边框可见
    ax2.spines['right'].set_visible(True)  # 右侧边框可见

    # 可选：调整边框的颜色和宽度
    ax2.spines['top'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')

    ax2.spines['top'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)
    ax2.spines['left'].set_linewidth(1)
    ax2.spines['right'].set_linewidth(1)
    # 合并所有图例元素
    all_lines = line1 + line2 + additional_lines + [bars]
    labels = [l.get_label() for l in all_lines]

    # 添加图例（调整位置防止溢出）
    ax2.legend(
        all_lines,
        labels,
        loc='upper right',
        bbox_to_anchor=(1, 1),  # 根据实际情况调整位置
        framealpha=0.9
    )
    # 设置公共参数
    ax.set_xlabel(f'时间步（间隔{time_step}）')
    plt.grid(False)


def plot_multiple_inference_lft(rain_fall, pred, ground_truth, iter_,
                            pred_np1=None, pred_np2=None, pred_np3=None,  # 新增多个预测结果参数
                            pred_np4=None, pred_np5=None, idx_list=None,
                            figures_folder='figures_jhhhhc_r10',
                            time_step='10 min', save=False):
    """支持多预测结果的综合可视化"""
    # 转换为numpy数组（兼容None值）
    rain_fall_np = rain_fall.cpu().numpy()
    pred_np = pred.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    # 转换额外预测数据（保留None）
    def tensor_to_np(x):
        return x.cpu().numpy() if x is not None else None

    pred_np1 = tensor_to_np(pred_np1)
    pred_np2 = tensor_to_np(pred_np2)
    pred_np3 = tensor_to_np(pred_np3)
    pred_np4 = tensor_to_np(pred_np4)
    pred_np5 = tensor_to_np(pred_np5)

    # 创建画布和子图
    fig, axs = plt.subplots(3, 2, figsize=(18, 15))

    # 默认绘制前6个样本
    idx_list = idx_list if idx_list else list(range(6))
    assert len(idx_list) == 6, "需要6个样本索引"

    # 统一调用新绘图函数
    for i, ax in enumerate(axs.flat):
        plot_single_inference_lft(
            ax=ax,
            rain_fall_np=rain_fall_np,
            pred_np=pred_np,
            pred_np1=pred_np1,
            pred_np2=pred_np2,
            pred_np3=pred_np3,
            pred_np4=pred_np4,
            pred_np5=pred_np5,
            gt_np=gt_np,
            idx=idx_list[i],
            time_step=time_step
        )
        ax.set_xlim(0, None)  # 统一X轴起点

    # 调整布局与保存
    plt.tight_layout()
    if save:
        os.makedirs(figures_folder, exist_ok=True)
        plt.savefig(f"{figures_folder}/combined_plot_{iter_}.svg", bbox_inches='tight')
    plt.show()
    plt.close()


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_four_inference(rain_fall, pred, ground_truth, iter_, idx_list=None,
                            figures_folder='figures_jhhhhc_r10', time_step='10 min',
                            save=False):
    """绘制6个子图的综合可视化"""
    # 转换为numpy数组
    rain_fall_np = rain_fall.cpu().numpy()
    pred_np = pred.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    # 创建画布和子图
    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    # fig.suptitle(f'Model Predictions vs Ground Truth (Iteration {iter_})',
    #              fontsize=16, y=1.02)

    # 默认绘制前6个样本
    if idx_list is None:
        idx_list = list(range(4))
    else:
        assert len(idx_list) == 4, "需要4个索引值"

    # 遍历每个子图进行绘制
    for i, ax in enumerate(axs.flat):
        if rain_fall_np.shape[1] == pred_np.shape[1]:
            plot_single_inference_td(ax, rain_fall_np, pred_np, gt_np, idx_list[i], time_step)
        else:
            plot_single_inference_pre(ax, rain_fall_np, pred_np, gt_np, idx_list[i], time_step)
        ax.set_xlim(left=0)  # 统一X轴起点

    # 调整布局
    plt.tight_layout()

    # 保存结果
    if save:
        os.makedirs(figures_folder, exist_ok=True)
        filename = os.path.join(figures_folder, f'combined_plot_{iter_}.svg')
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to {filename}')
    plt.grid(False)
    plt.show()
    plt.close()


