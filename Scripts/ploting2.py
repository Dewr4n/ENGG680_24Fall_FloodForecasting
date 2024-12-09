import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data_file = "mean_result.csv"
data = pd.read_csv(data_file)

# 指定列索引和时间列
columns_to_plot = [5, 7, 11, 12, 13]
time_column = 0  # 时间列索引

# 提取时间数据
time_data = data.iloc[:, time_column].str[:10]  # 截取前10个字符，只保留日期部分
selected_ticks = ["2019-10-31", "2020-10-31", "2021-10-31", "2022-10-31", "2023-10-31", "2024-10-31"]

# 绘图
for col in columns_to_plot:
    plt.figure(figsize=(10, 6))

    # 提取对应列数据
    y_data = data.iloc[:, col]
    y_label = data.columns[col]  # 使用列标题作为Y轴标签

    # 绘制线性图
    plt.plot(time_data, y_data, linestyle='-', label=y_label, linewidth=0.5)

    # 设置标题和坐标轴标签
    plt.title(f"{y_label} over time", fontsize=14)
    plt.ylabel(y_label, fontsize=12)

    # 调整x轴刻度以适应时间数据
    plt.xticks(selected_ticks, rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图像
    output_file = f"output/plots/line_plot_column_{col}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Plot for column {col} saved as {output_file}.")
