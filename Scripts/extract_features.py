import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取CSV文件
file_path = "mean_result.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 提取最后一列数据
last_column = data.iloc[:, -1]  # 获取最后一列
flood_count = (last_column == 1).sum()  # 统计1的数量
no_flood_count = (last_column == -1).sum()  # 统计-1的数量

# 创建柱状图数据
categories = ['Flood', 'No Flood']
counts = [flood_count, no_flood_count]

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['red', 'blue'], alpha=0.7)
plt.title('Flood and No Flood Hourly Distribution for Data', fontsize=16)
plt.ylabel('Hours', fontsize=14)

# 添加数据标签
for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.tight_layout()

# 保存图像
output_dir = "output"  # 图像保存的文件夹
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建
output_path = os.path.join(output_dir, "flood_distribution.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"柱状图已保存为：{output_path}")
