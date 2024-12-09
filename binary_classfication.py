import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os  # 用于文件操作
import time  # 用于生成时间戳
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width

# 添加模型保存路径
model_save_path = "models/best.pth"

"""
# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""

# 获取和转换数据
def get_data(file_path, start_time, end_time):
    """
    根据实际数据获取训练数据，仅使用第5、第8、第11列作为特征，并根据第一列的时间范围筛选行。

    参数:
        file_path: str, CSV 文件的路径。
        start_time: str, 起始时间，例如 '2020-06-30 00:00'。
        end_time: str, 结束时间，例如 '2020-09-30 23:00'。

    返回:
        x_tensor: torch.Tensor, 特征的张量。
        y_tensor: torch.Tensor, 标签的张量。
    """

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        """
        将 pandas 的列或 dataframe 转为 PyTorch 的 tensor。
        """
        return torch.tensor(data=dataframe_series.values)

    # 读取数据
    data = pd.read_csv(file_path)

    # 将第一列解析为 datetime 格式
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])

    # 将起始和结束时间转换为 Timestamp 类型
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # 筛选时间范围的行
    filtered_data = data[(data.iloc[:, 0] >= start_time) & (data.iloc[:, 0] <= end_time)]
    # 特征数据 (第5、第8、第11列)
    feature_columns = [5, 7, 11, 12, 13]   # 按索引提取特定列
    x = filtered_data.iloc[:, feature_columns]

    # 标签数据 (第18列)，将 -1 替换为 0
    y = filtered_data.iloc[:, 17].replace(-1, 0)

    # 将特征进行标准化处理
    x_scaled = preprocessing.StandardScaler().fit_transform(x)
    x_df = pd.DataFrame(data=x_scaled)

    # 转换为 PyTorch Tensor 格式
    x_tensor = get_tensor_from_pd(x_df).float()
    y_tensor = get_tensor_from_pd(y).float()

    Time = filtered_data.iloc[:, 0]
    return x_tensor, y_tensor , Time

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=100, output_size=1):
        """
        LSTM for binary classification
        :param input_size
        :param hidden_layer_size
        :param output_size
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, dropout=0.4)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        hidden_cell = (torch.zeros(2, input_x.size(1), self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(2, input_x.size(1), self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(linear_out)
        return predictions


if __name__ == '__main__':
    # 保存图表的路径
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Get the training data
    x, y , Time_train= get_data('mean_result.csv',
                                '2023-10-31 00:00',
                                '2024-10-31 00:00')
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),
        batch_size=120,
        shuffle=False,
        num_workers=2,
    )
    # loss，optimization，epochs
    model = LSTM()  #model initialization
    loss_function = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001)  # optimizer
    epochs = 500

    # 用于记录每次训练的loss
    train_losses = []

    # 检查是否已有保存的模型
    if os.path.exists(model_save_path):
        print(f"Found saved model at {model_save_path}, loading it...")
        model = LSTM()
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
    else:
        # 训练阶段
        print("No saved model found, training a new model...")
        model.train()
        best_loss = float("inf")  # 初始化最佳损失值
        best_model_state = None  # 用于存储最佳模型状态
        for i in range(epochs):
            epoch_loss = 0  # 初始化 epoch 损失
            for seq, labels in train_loader:
                optimizer.zero_grad()
                y_pred = model(seq).squeeze()
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
                epoch_loss += single_loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            print(f"Epoch {i + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

            # 更新最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = model.state_dict()

            # 早停条件
            if avg_epoch_loss < 0.05:
                print(f"Stopping early at Epoch {i + 1} as Average Loss < 0.05")
                break

        # 检查并创建保存目录
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))

        # 保存最终的最佳模型
        if best_model_state is not None:
            torch.save(best_model_state, model_save_path)
            print(f"Best model saved with loss {best_loss:.4f}")

        # 绘制训练损失的变化图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, color='blue', label='Training Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training Loss (BCE)', fontsize=16)
        plt.savefig(os.path.join(output_dir, f"Training_Loss_{int(time.time())}.png"))
        plt.grid(True)
        plt.show()

    # 进行测试
    x_test, y_test, Time_test = get_data('mean_result.csv', '2021-06-30 00:00', '2021-11-30 00:00')

    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x_test, y_test),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=120,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    epoch_loss = 0.0
    y_true = []
    y_pred_list = []
    y_pred_raw = []

    with torch.no_grad():
        for seq, labels in test_loader:
            y_pred = model(seq).squeeze()
            single_loss = loss_function(y_pred, labels)
            epoch_loss += single_loss.item()

            y_true.extend(labels.cpu().numpy())
            y_pred_list.extend(torch.where(y_pred > 0.3, 1, 0).detach().cpu().numpy())
            y_pred_raw.extend(y_pred.detach().cpu().numpy())

    avg_epoch_loss = epoch_loss / len(test_loader)
    print(f"Average Test Loss: {avg_epoch_loss:.4f}")

    # 计算额外的评估指标
    accuracy = accuracy_score(y_true, y_pred_list)
    precision = precision_score(y_true, y_pred_list)
    recall = recall_score(y_true, y_pred_list)
    f1 = f1_score(y_true, y_pred_list)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # 日志记录
    log_path = "training_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"Test Loss: {avg_epoch_loss:.4f}\\n")
        log_file.write(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\\n")
        log_file.write("-" * 30 + "\\n")


    # 绘制图表
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_raw, bins=50, kde=True, color='blue')
    plt.title('Distribution of Flood Probability', fontsize=16)
    plt.xlabel('Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"flood_probability_distribution_{int(time.time())}.png"))
    plt.show()

    # 绘制时间序列图，只显示特定的行（第一行和最后一行的时间标签）
    plt.figure(figsize=(14, 7))
    # plt.plot(range(len(y_pred_raw)), y_pred_raw, linestyle='-', color='blue', label='Flood Probability')
    plt.plot(Time_test, y_pred_raw, linestyle='-', color='blue', label='Flood Probability')
    plt.axhline(y=0.3, color='red', linestyle='--', label='Warning Threshold')
    # 添加标签 '0.3' 刚好在红线的上方
    x_center = (plt.xlim()[0] + plt.xlim()[1]) / 2
    plt.text(x_center, 0.2, '0.2', color='red', fontsize=12, ha='center')
    plt.title('Flood Probability Over Time', fontsize=16)
    # plt.xlabel('Time', fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('Probability', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"flood_probability_over_time_{int(time.time())}.png"))
    plt.show()

    ######Real Label#######
    plt.figure(figsize=(14, 7))
    plt.plot(Time_test, y_test, linestyle='-', color='green', label='GroundTruth')
    plt.plot(Time_test, y_pred_raw, linestyle='--', color='blue', label='Prediction')
    plt.axhline(y=0.3, color='red', linestyle='--', label='Warning Threshold')
    # 添加标签 '0.3' 刚好在红线的上方
    x_center = (plt.xlim()[0] + plt.xlim()[1]) / 2
    plt.text(x_center, 0.3, '0.3', color='red', fontsize=12, ha='center')
    plt.title('Prediction vs GroundTruth', fontsize=16)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('Flood Probability', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(output_dir, f"prediction_label_{int(time.time())}.png"))
    plt.show()

    ######Combine#######
    plt.figure(figsize=(14, 7))
    plt.plot(Time_test, y_test, linestyle='-', linewidth=2.5 ,color='green', label='GroundTruth')
    plt.plot(Time_test, y_pred_raw, linestyle='-', color='blue', label='Prediction')
    plt.title('Prediction vs GroundTruth', fontsize=16)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('Flood Probability', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(output_dir, f"prediction_label_{int(time.time())}.png"))
    plt.show()


    cm = confusion_matrix(y_true, y_pred_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood', 'Flood'],
                yticklabels=['No Flood', 'Flood'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{int(time.time())}.png"))
    plt.show()