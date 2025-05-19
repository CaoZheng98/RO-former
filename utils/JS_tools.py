import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import seaborn as sns
# 指定支持中文的字体
sns.set(font='Microsoft YaHei')
backend_inline.set_matplotlib_formats('svg')
backend_inline.set_matplotlib_formats('png')
from pathlib import Path
import os
import numpy as np


def generate_segments(df, tgt_len):
    # 计算索引之间的时间差，并找出大于5分钟的地方
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)].index
    of_idx_lst = [0]
    # 如果gaps列表为空，说明没有找到大于5分钟的间隔
    if gaps.empty:
        print("没有找到间隔")
    else:
        # 找出每个连续段的开始和结束
        segments = []
        of_sample_lst = []
        start_idx_pos = 0
        start_idx = df.index[start_idx_pos]
        for gap in gaps:
            # 使用位置索引来获取结束索引
            end_idx_pos = df.index.get_loc(gap) - 1
            end_idx = df.index[end_idx_pos]
            num = end_idx_pos - start_idx_pos + 1
            if tgt_len <= num:
                segments.append(df.loc[start_idx:end_idx].values.astype("float32"))
                v = num - tgt_len + 1
                of_sample_lst.append(v)
            # 更新下一个开始索引
            start_idx_pos = df.index.get_loc(gap)
            start_idx = df.index[start_idx_pos]
        # 添加最后一段
        # segments.append((start_idx, df.index[-1]))
        end_idx_pos = len(df.index) - 1
        end_idx = df.index[end_idx_pos]
        num = end_idx_pos - start_idx_pos + 1
        if tgt_len <= num:
            segments.append(df.loc[start_idx:end_idx].values.astype("float32"))
            v = num - tgt_len + 1
            of_sample_lst.append(v)

    for of_len in of_sample_lst:
        v = of_len + of_idx_lst[-1]
        of_idx_lst.append(v)
    return segments, of_idx_lst

def process1_data(df, time_step):
    # 计算实时流量和降雨量
    df[f'实时流量(m³/{time_step})'] = df['累计流量(m³)'].diff()
    df[f'实时降雨量(mm/{time_step})'] = df['年度累计降雨量(mm)'].diff()
    df[f'实时流量(m³/{time_step})'] = df[f'实时流量(m³/{time_step})'].apply(lambda x: max(x, 0))
    df[f'实时降雨量(mm/{time_step})'] = df[f'实时降雨量(mm/{time_step})'].apply(lambda x: max(x, 0))
    # 删除累计流量和降雨量列
    df = df.drop(columns=['累计流量(m³)', '年度累计降雨量(mm)', '当天累计降雨量(mm)', ])
    return df
def freq(df):
    # 计算时间间隔
    df.loc[:, '时间间隔'] = df.index.to_series().diff().dt.total_seconds() / 60
    # 统计时间间隔的分布
    interval_counts = df['时间间隔'].value_counts().sort_index()
    # 将类别和占比数据单独列出
    category_data = pd.DataFrame({
        '时间间隔 (分钟)': interval_counts.index.astype(str),
        '频次': interval_counts.values,
        '占比 (%)': interval_counts.values / interval_counts.sum() * 100
    })
    # 打印类别和占比数据
    # messgae = {}
    # for index, row in category_data.iterrows():
    #     messgae[f"时间间隔: {row['时间间隔 (分钟)']}"] = f"频次: {int(row['频次'])}, 占比: {row['占比 (%)']:.2f}%"
    # 找到占比最大的索引
    max_index = category_data['占比 (%)'].idxmax()

    # 使用找到的索引来获取整行数据
    max_row = category_data.loc[max_index]
    print(f"最多间隔: {max_row['时间间隔 (分钟)']}  频次: {int(max_row['频次'])}, 占比: {max_row['占比 (%)']:.2f}% /n")
    return max_row

def draw_df(df, save_name=None):
    n = len(df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(18, 20))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        axes[i].scatter(df.index, df[col],s = 5 )
        axes[i].set_title(col)
        axes[i].set_xlabel('date')
        axes[i].set_ylabel(col)
        axes[i].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    if save_name is not None:
        save_path = 'C:/Users/34938/Desktop'
        plt.savefig(os.path.join(save_path,f'{save_name}.png'), dpi=1200,facecolor=plt.gcf().get_facecolor())
    plt.show()


def ncon(df):
    # 计算索引之间的时间差，并找出大于5分钟的地方
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)].index

    # 如果gaps列表为空，说明没有找到大于5分钟的间隔
    if gaps.empty:
        print("没有找到大于5分钟的间隔")
    else:
        # 找出每个连续段的开始和结束
        segments = []
        start_idx = df.index[0]
        for gap in gaps:
            # 使用位置索引来获取结束索引
            end_idx_pos = df.index.get_loc(gap) - 1
            end_idx = df.index[end_idx_pos]
            segments.append((start_idx, end_idx))
            # 更新下一个开始索引
            start_idx_pos = df.index.get_loc(gap)
            start_idx = df.index[start_idx_pos]
        # 添加最后一段
        segments.append((start_idx, df.index[-1]))

        # 打印连续段的开始和结束
        print("连续段的开始和结束索引:")
        for start, end in segments:
            if end - start > pd.Timedelta(days=10):
                print(f"开始: {start}, 结束: {end}， 持续时间：{end - start}\n")
        # 打印间隔数量
        print(f"间隔数量（段数）: {len(segments)}\n")

def demo_n(df, tgt_len, of_idx):
    temp_lst = df.values.tolist()
    # 计算索引之间的时间差，并找出大于5分钟的地方
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)].index

    # 如果gaps列表为空，说明没有找到大于5分钟的间隔
    if gaps.empty:
        print("没有找到大于5分钟的间隔")
    else:
        # 找出每个连续段的开始和结束
        segments = []
        local_idx_lst = []
        start_idx_pos = 0
        start_idx = df.index[start_idx_pos]
        sum = 0
        for gap in gaps:
            # 使用位置索引来获取结束索引
            end_idx_pos = df.index.get_loc(gap) - 1
            end_idx = df.index[end_idx_pos]
            num = end_idx_pos - start_idx_pos + 1
            if tgt_len <= num:
                local_idx_lst += list(range(start_idx_pos, end_idx_pos + 1))
                sum += num - tgt_len + 1
            # 更新下一个开始索引
            start_idx_pos = df.index.get_loc(gap)
            start_idx = df.index[start_idx_pos]
        # 添加最后一段
        end_idx_pos = len(df.index) - 1
        end_idx = df.index[end_idx_pos]
        num = end_idx_pos - start_idx_pos + 1
        if tgt_len <= num:
            local_idx_lst += list(range(start_idx_pos, end_idx_pos + 1))
            segments.append((start_idx, end_idx))
    return sum, local_idx_lst
def load_forcing_data():
    project_root = Path(os.getcwd()).parent
    data_path = project_root / "JieShou/gauge"
    gauge_names = ['HMG', 'XFG']
    out_put_label = ['采集时间', '大气温度(℃)', '大气湿度(%)', '风速(m/s)', '年度累计降雨量(mm)', '当天累计降雨量(mm)',
                     '风向(°)']
    out_put_step = None  # 'D'
    gauge_dic = {}
    for gauge_name in gauge_names:
        files = list(data_path.glob(f"{gauge_name}*.xls"))
        dfs = []
        for file in files:
            df = pd.read_excel(file)
            dfs.append(df)
        if dfs:
            temp_df = pd.concat(dfs, ignore_index=True)
            # 将第一列转换为datetime格式
            temp_df['采集时间'] = pd.to_datetime(temp_df['采集时间'], format='%Y/%m/%d %H:%M', errors='coerce')
            # 按时间排序
            temp_df = temp_df.sort_values(by='采集时间').reset_index()
            # 去除第一列不是时间所在行的数据
            temp_df = temp_df[temp_df['采集时间'].dt.year > 1900]
            # 过滤不需要输出的列
            temp_df = temp_df.loc[:, out_put_label]
            # 将 '采集时间' 列设置为索引
            temp_df.set_index('采集时间', inplace=True)
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            temp_df.index.name = None
            gauge_dic[gauge_name] = temp_df

    # 两个气象站的数据相互补齐
    date1s = gauge_dic['HMG'].index[0]  # pd.to_datetime('2022-07-01 00:02:00')
    date1e = gauge_dic['XFG'].index[0]  # pd.to_datetime('2022-07-18 15:58:00')
    date2s = gauge_dic['XFG'].index[-1]  # pd.to_datetime('2023-06-29 01:43:00')
    date2e = gauge_dic['HMG'].index[-1]  # pd.to_datetime('2024-04-17 22:15:00')
    df_HMG = gauge_dic['HMG']
    segments = {}
    # 选择第一段
    segments['segment1'] = df_HMG[(df_HMG.index >= date1s) & (df_HMG.index <= date1e)]
    # 选择第二段
    # 这里假设date2s是第二段的开始，如果需要包括date1e，可以将条件改为df_HMG.index > date1e
    segments['segment2'] = df_HMG[(df_HMG.index > date2s) & (df_HMG.index <= date2e)]
    df_XFG = gauge_dic['XFG']
    df_temp = df_XFG.copy()
    for segment in segments.values():
        # 合并索引
        combined_index = df_temp.index.union(segment.index)
        # 对齐索引
        df_temp_aligned = df_temp.reindex(combined_index)
        segment_aligned = segment.reindex(combined_index)
        # 使用 combine_first 相互补齐缺失值
        df_temp = df_temp_aligned.combine_first(segment_aligned)
    return df_temp

def load_outfall():
    project_root = Path(os.getcwd()).parent
    data_path = project_root / "JieShou/outfall"
    outfall_names = ['DCHTLN', 'DCHWSC', 'HMGDX', 'HMGSM', 'HMGWS', 'JBHDT', 'JBHWZ', 'JHHHHC', 'JHHXY', 'JLH', 'WFG',
                     'XFG']
    out_put_label = ['采集时间', '溶解氧(mg/L)', '氨氮(mg/L)', '累计流量(m³)']
    # out_put_step = 'D'
    outfall_dic = {}
    for outfall_name in outfall_names:
        files = list(data_path.glob(f"{outfall_name}*.xls"))
        dfs = []
        for file in files:
            df = pd.read_excel(file)
            dfs.append(df)
        if dfs:
            temp_df = pd.concat(dfs, ignore_index=True)
            # 将第一列转换为datetime格式
            temp_df['采集时间'] = pd.to_datetime(temp_df['采集时间'], format='%Y/%m/%d %H:%M', errors='coerce')
            # 按时间排序
            temp_df = temp_df.sort_values(by='采集时间').reset_index()
            # 去除第一列不是时间所在行的数据
            temp_df = temp_df[temp_df['采集时间'].dt.year > 1900]
            # 过滤不需要输出的列
            temp_df = temp_df.loc[:, out_put_label]
            # if out_put_step == 'D':
            #     temp_df.set_index('采集时间', inplace=True)
            #     # 按天进行分组并计算每天的平均值和最大值
            #     temp_df = temp_df.groupby(temp_df.index.date).agg({
            #         '溶解氧(mg/L)': 'mean',
            #         '氨氮(mg/L)': 'mean',
            #         '当天累计流量(m³)': 'max'
            #     }).rename(columns={'溶解氧(mg/L)': '日平均溶解氧(mg/L)',
            #                        '氨氮(mg/L)': '日平均氨氮(mg/L)',
            #                        '当天累计流量(m³)': '最大当天累计流量(m³)'})
            temp_df.set_index('采集时间', inplace=True)
            temp_df.index.name = None
            outfall_dic[outfall_name] = temp_df
    return outfall_dic


column_mapping = {
    'D_DATETIME': '资料时间',
    'V01301': '台站号',
    'V04001': '年',
    'V04002': '月',
    'V04003': '日',
    'V05001': '纬度',
    'V06001': '经度',
    'V07001': '测站高度',
    'V13011': '日累计降水量',
    'Q13011': '日降水量质控码',
    'V12001': '日平均气温',
    'Q12001': '日平均气温质控码'
}

def load_cma():
    project_root = Path(os.getcwd()).parent
    data_path = [project_root / "JieShou/cma/precipitation", project_root / "JieShou/cma/temp"]
    metro_name = ['SURF_GLB_PRE_DAY_V2_Asia', 'SURF_GLB_TEM_DAY_PROD_Asia']
    forcing_datas = dict()
    for i in range(len(data_path)):
        files = list(data_path[i].glob(f"**/{metro_name[i]}_*.txt"))
        dfs = []
        for file in files:
            df = pd.read_csv(file, sep=r"\s+")
            df_renamed = df.rename(columns=column_mapping)
            # 筛选出阜阳的数据
            df_filtered = df_renamed[df_renamed['台站号'].str.endswith('58203')]
            dfs.append(df_filtered)
        # 使用concat合并所有DataFrame
        df_combined = pd.concat(dfs, ignore_index=True)
        dates = df_combined['年'].map(str) + "/" + df_combined['月'].map(str) + "/" + df_combined['日'].map(str)
        df_combined.index = pd.to_datetime(dates, format="%Y/%m/%d")
        if metro_name[i] == 'SURF_GLB_PRE_DAY_V2_Asia':
            # 保留特定的列
            df_combined = df_combined.loc[:, ['日累计降水量']]  # , '日平均气温质控码'
        elif metro_name[i] == 'SURF_GLB_TEM_DAY_PROD_Asia':
            df_combined = df_combined.loc[:, ['日平均气温']]  # , '日平均气温质控码'
        forcing_datas[metro_name[i]] = df_combined

    base_df = forcing_datas[metro_name[0]].reset_index()
    for metro in metro_name[1:]:
        base_df = pd.merge(base_df, forcing_datas[metro].reset_index(), on='index', how='outer')
    base_df.set_index('index', inplace=True)
    base_df.index.name = None
    base_df = base_df[~base_df.index.duplicated(keep='first')]
    return base_df

def _process_invalid_data(df: pd.DataFrame):
    # Delete all row, where exits NaN (only discharge has NaN in this dataset)
    len_raw = len(df)
    df = df.dropna()
    len_drop_nan = len(df)
    if len_raw > len_drop_nan:
        print(f"Deleted {len_raw - len_drop_nan} records because of NaNs {outfall}")

    # Deletes all records, where no discharge was measured (-999)
    df = df.drop((df[df['最大当天累计流量(m³)'] < 0]).index)
    len_drop_neg = len(df)
    if len_drop_nan > len_drop_neg:
        print(f"Deleted {len_drop_nan - len_drop_neg} records because of negative discharge {outfall}")

    return df

def calc_mean_and_std(data_dict):
    data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
    nan_mean = np.nanmean(data_all, axis=0)
    nan_std = np.nanstd(data_all, axis=0)
    return nan_mean, nan_std

# def _local_normalization( feature: np.ndarray, variable: str) -> np.ndarray:
#     if variable == 'inputs':
#         feature = (feature - x_mean) / x_std
#     elif variable == 'output':
#         feature = (feature - y_mean) / y_std
#     else:
#         raise RuntimeError(f"Unknown variable type {variable}")
#     return feature

def re_sample_of(outfall_data, time_step):

    df = outfall_data
    temp = []
    for column in df.columns:
        # 对'rainfall'列应用每5分钟取最大值的重采样策略
        if '流量' in column:
            temp_df = df[column].resample(time_step).max()
        else:
            temp_df = df[column].resample(time_step).mean()
        temp.append(temp_df)
    of_r = pd.concat(temp, axis=1)
    return of_r

def re_sample_fd(df, time_step):
    temp = []
    for column in df.columns:
        # 对'rainfall'列应用每5分钟取最大值的重采样策略
        if '降雨' in column:
            temp_df = df[column].resample(time_step).max()
        else:
            temp_df = df[column].resample(time_step).mean()
        temp.append(temp_df)
    df_resampled = pd.concat(temp, axis=1)
    return df_resampled


import pandas as pd
import seaborn as sns
import pandas
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    计算DataFrame的相关性矩阵并绘制热图。

    参数:
    df -- 输入的Pandas DataFrame
    """
    # 确保输入是DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是Pandas DataFrame")

    # 计算相关性矩阵
    corr_matrix = df.corr()

    # 创建一个热图
    plt.figure(figsize=(10, 8))  # 可以根据需要调整大小
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()


import pandas as pd
import math


def are_in_same_order_of_magnitude(num1, num2, threshold=2):
    """
    判断两个数是否在同一个数量级。

    :param num1: 第一个数
    :param num2: 第二个数
    :param threshold: 数量级差异的阈值，默认为1
    :return: 如果两个数在同一个数量级返回True，否则返回False
    """
    # 检查输入是否为正数，因为对数函数只对正数定义
    num1 += 0.0001
    num2 += 0.0001
    if num1 < 0 or num2 < 0:
        raise ValueError("输入的数必须是正数")

    # 计算两个数的对数
    log_num1 = math.log10(abs(num1))
    log_num2 = math.log10(abs(num2))

    # 计算对数之差
    log_diff = abs(log_num1 - log_num2)

    # 如果对数之差小于阈值，则认为在同一个数量级
    return log_diff < threshold


def find_rain_events_with_context(df, outfall_mask,threshold,time_step):
    # 确保 '实时降雨量(mm/h)' 列是数值类型
    df[f'实时降雨量(mm/{time_step})'] = pd.to_numeric(df[f'实时降雨量(mm/{time_step})'], errors='coerce')
    # 确保 '实时流量(m³/h)' 列是数值类型
    df[f'实时流量(m³/{time_step})'] = pd.to_numeric(df[f'实时流量(m³/{time_step})'], errors='coerce')

    # 初始化结果字典
    rain_events = []
    event_index = 0  # 用于编号降雨事件
    start_time = None
    end_time = None
    event_indices = []
    start_flow = None  # 记录降雨开始前的流量

    # 遍历DataFrame
    for i, row in df.iterrows():
        if row[f'实时降雨量(mm/{time_step})'] > 0:
            if start_time is None:
                # 如果当前是第一个非零降雨量，标记为事件开始
                start_time = row.name
                end_time = row.name
                event_indices = [i]
                # 记录降雨开始前的流量
                start_time_idx = df.index.get_loc(start_time)
                if start_time_idx > 3:  # 确保有足够的数据来计算流量
                    start_flow = df[f'实时流量(m³/{time_step})'].iloc[start_time_idx - 3:start_time_idx].mean()
                else:
                    start_flow = df[f'实时流量(m³/{time_step})'].iloc[start_time_idx]
            else:
                # 如果已经在事件中，更新结束时间并添加索引
                end_time = row.name

        else:
            if start_time is not None:
                # 如果当前是零降雨量且之前有非零降雨量，检查是否结束事件
                event_end_idx = df.index.get_loc(end_time)
                # 检查前后3小时内是否有其他非零降雨量
                start_idx = max(0, df.index.get_loc(start_time) - 3)
                end_idx = event_end_idx
                patience = 0
                while not are_in_same_order_of_magnitude(df[f'实时流量(m³/{time_step})'].iloc[end_idx], start_flow, threshold=threshold):
                    if (df.iloc[start_idx:end_idx + 1][f'实时降雨量(mm/{time_step})'] > 0).sum() == len(event_indices):
                        end_idx += 1
                        patience += 1
                        if patience > 24:
                            print(
                                f"警告：outfall_mask为{outfall_mask}的第{event_index}场降雨的流量不能在预期的耐心时间内恢复到降雨前的数量级,其start_idx为{start_idx}")
                            break
                    else:
                        patience = 0
                        end_idx += 1
                        # 选择子集
                        temp = df.iloc[start_idx:end_idx + 1]

                        # 将 'outfall_mask' 列插入到倒数第二列
                        # len(temp.columns) 是列的总数，-2 表示倒数第二列的位置
                        temp.insert(loc=len(temp.columns) - 1, column='outfall_mask', value=outfall_mask)
                        temp = temp.to_numpy(dtype='float32')
                        # 将修改后的temp添加到rain_events列表
                        rain_events.append(temp)

                event_index += 1

                # 重置开始时间和事件索引列表
                start_time = None
                end_time = None
                event_indices = []
                start_flow = None

    # 检查是否有未结束的事件（即最后一段降雨）
    if start_time is not None:
        event_end_idx = len(df) - 1
        start_idx = max(0, df.index.get_loc(start_time) - 3)
        end_idx = event_end_idx
        patience = 0
        while not are_in_same_order_of_magnitude(df[f'实时流量(m³/{time_step})'].iloc[end_idx], start_flow, threshold=threshold):
            if (df.iloc[start_idx:end_idx + 1][f'实时降雨量(mm/{time_step}'] > 0).sum() == len(event_indices):
                end_idx += 1
                if end_idx == df.shape[0]:
                    end_idx -= 1
                    break
                patience += 1
                if patience > 3:
                    print(f"警告：outfall_mask为{outfall_mask}的第{event_index}场降雨的流量不能在预期的耐心时间内恢复到降雨前的数量级,其start_idx为{start_idx}")
                    break
            else:
                patience = 0
                end_idx += 1
                if end_idx == df.shape[0]:
                    end_idx -= 1
                    break
        # 选择子集
        temp = df.iloc[start_idx:end_idx + 1]

        # 将 'outfall_mask' 列插入到倒数第二列
        # len(temp.columns) 是列的总数，-2 表示倒数第二列的位置
        temp.insert(loc=len(temp.columns) - 1, column='outfall_mask', value=outfall_mask)
        temp = temp.to_numpy(dtype='float32')
        # 将修改后的temp添加到rain_events列表
        rain_events.append(temp)

    return rain_events