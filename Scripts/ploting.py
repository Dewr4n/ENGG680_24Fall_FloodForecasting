import matplotlib.pyplot as plt
import numpy as np

# 从文件中读取数据
data_file = "output/plots/FloodTimes.txt"
years = []
duration_days = []

with open(data_file, "r") as file:
    for line in file:
        parts = line.strip().split()
        year = int(parts[0])
        days = int(parts[1])
        years.append(year)
        duration_days.append(days)

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制直方图
bars = plt.bar(years, duration_days, color='skyblue', edgecolor='black', alpha=0.8)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f'{int(height)} days', ha='center', va='bottom', fontsize=10)

# 设置标题和坐标轴标签
plt.title("Flood Duration in Kampala by Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Flood Duration", fontsize=12)

# 美化x轴
plt.xticks(years)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图形
output_file = "output/plots/Flood_Duration_Kampala.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Histogram saved as {output_file}")

# 显示图形
plt.show()