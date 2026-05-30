"""
项目2：图像分类 - CIFAR-10 + CNN
基于 PaddlePaddle 官方示例代码
"""

import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
import numpy as np
import time

print('PaddlePaddle version:', paddle.__version__)
paddle.set_device('cpu')

# ── 加载数据集 ────────────────────────────────────────────────────
print('\n加载 CIFAR-10 数据集...')

transform = ToTensor()
cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform)

print(f'训练集: {len(cifar10_train)} 张 | 测试集: {len(cifar10_test)} 张')

# ── 定义网络 ──────────────────────────────────────────────────────
class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# ── 训练配置 ──────────────────────────────────────────────────────
epoch_num = 10
batch_size = 32
learning_rate = 0.001
val_acc_history = []
val_loss_history = []

def train(model):
    print('\n' + '='*60)
    print('开始训练（10 epochs, batch_size=32）...')
    print('优化器: Adam (learning_rate=0.001)')
    print('='*60 + '\n')

    start_time = time.time()

    model.train()

    opt = paddle.optimizer.Adam(learning_rate=learning_rate,
                                parameters=model.parameters())

    train_loader = paddle.io.DataLoader(cifar10_train,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)

    for epoch in range(epoch_num):
        epoch_start = time.time()

        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)

            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(loss.numpy())))

            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        epoch_time = time.time() - epoch_start

        print("[validation] epoch {} - accuracy: {:.4f}, loss: {:.4f}, time: {:.1f}s".format(
            epoch, avg_acc, avg_loss, epoch_time))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()

    total_time = time.time() - start_time
    print('\n训练完成！总耗时: {:.1f} 分钟'.format(total_time / 60))

    return avg_acc, avg_loss

# ── 开始训练 ──────────────────────────────────────────────────────
model = MyNet(num_classes=10)
final_acc, final_loss = train(model)

# ── 最终测试评估 ───────────────────────────────────────────────────
print('\n' + '='*60)
print('最终测试集评估...')
print('='*60)

model.eval()
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)

all_preds = []
all_targets = []

print('预测测试集...')
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = paddle.to_tensor(data[1])
    y_data = paddle.unsqueeze(y_data, 1)

    logits = model(x_data)
    preds = paddle.argmax(logits, axis=1)

    all_preds.extend(preds.numpy().tolist())
    all_targets.extend(y_data.numpy().flatten().tolist())

    if (batch_id + 1) % 100 == 0:
        print(f'  已预测 {(batch_id + 1) * batch_size}/{len(cifar10_test)} 张图片...')

print(f'预测完成！共 {len(all_preds)} 个样本')

# ── 计算 Accuracy, Recall, Precision ─────────────────────────────
correct = sum(p == t for p, t in zip(all_preds, all_targets))
test_acc = correct / len(all_preds)
error_rate = 1.0 - test_acc

# 计算 Macro Recall 和 Macro Precision
from collections import defaultdict
tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

for p, t in zip(all_preds, all_targets):
    if p == t:
        tp[t] += 1
    else:
        fp[p] += 1
        fn[t] += 1

recalls, precisions = [], []
for c in range(10):
    r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
    p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
    recalls.append(r)
    precisions.append(p)

macro_recall = sum(recalls) / 10
macro_precision = sum(precisions) / 10

# ── 输出结果 ─────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  Accuracy:       {test_acc:.4f}')
print(f'  Error-rate:     {error_rate:.4f}')
print(f'  Macro Recall:   {macro_recall:.4f}')
print(f'  Macro Precision:{macro_precision:.4f}')
print(f'{"="*60}')

print('\n┌─────────────────────────────────────────────────────────┐')
print('│  项目2实验结果（可填入报告）                              │')
print('├─────────────────────────────────────────────────────────┤')
print(f'│  Accuracy:       {test_acc:.4f}  ({test_acc*100:.2f}%)                   │')
print(f'│  Error-rate:     {error_rate:.4f}  ({error_rate*100:.2f}%)                 │')
print(f'│  Macro Recall:   {macro_recall:.4f}                                      │')
print(f'│  Macro Precision:{macro_precision:.4f}                                      │')
print('└─────────────────────────────────────────────────────────┘')

# ── 保存结果 ─────────────────────────────────────────────────────
import json
import os

result = {
    'accuracy': float(test_acc),
    'error_rate': float(error_rate),
    'macro_recall': float(macro_recall),
    'macro_precision': float(macro_precision)
}

result_path = os.path.join('.', 'cifar_result.json')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f'\n结果已保存至: {os.path.abspath(result_path)}')

# ── 保存模型 ─────────────────────────────────────────────────────
paddle.save(model.state_dict(), 'cifar_cnn.pdparams')
print('模型已保存至: cifar_cnn.pdparams')
print('\n 项目2训练完成！')
# 不知道是不是框架问题，跑出来的模型准确率这么低