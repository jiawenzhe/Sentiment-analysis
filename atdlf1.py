import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 准备IMDb数据集
def load_imdb_data(data_dir, split='train'):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, split, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels


# 数据集路径
data_dir = 'E:/datasets/aclImdb/train/aclImdb_v1/aclImdb'
train_texts, train_labels = load_imdb_data(data_dir, 'train')
test_texts, test_labels = load_imdb_data(data_dir, 'test')


# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ATDLF模型
class ATDLFModel(nn.Module):
    def __init__(self, num_labels):
        super(ATDLFModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        logits = self.fc(dropout_output)
        return self.softmax(logits)


# 训练历史记录类
class TrainingHistoryPlotter:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []

    def record(self, epoch, train_loss, val_loss):
        """记录每个epoch的训练和验证损失"""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def plot(self):
        """绘制损失曲线图并保存为文件"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_loss, label="Train Loss", marker='o')
        plt.plot(self.epochs, self.val_loss, label="Validation Loss", marker='o')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_comparison.png')  # 保存损失曲线图
        plt.close()


# 训练函数
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, num_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in tqdm(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / num_examples, np.mean(losses)


# 评估函数
def eval_model(model, data_loader, loss_fn, device, num_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / num_examples, np.mean(losses)


# 主函数
def main():
    # 参数设置
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    NUM_LABELS = 2
    LEARNING_RATE = 2e-5

    # 创建历史记录类
    history_plotter = TrainingHistoryPlotter()

    # 数据集划分
    X_train, X_val, y_train, y_val = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATDLFModel(NUM_LABELS)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // EPOCHS, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    # 训练和评估
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                            len(X_train))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(X_val))
        print(f'Val   loss {val_loss} accuracy {val_acc}')

        # 记录训练和验证损失
        history_plotter.record(epoch + 1, train_loss, val_loss)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model_state.bin')

    print(f'Best val accuracy: {best_accuracy}')

    # 生成损失曲线图
    history_plotter.plot()

    # 测试预测
    model.load_state_dict(torch.load('best_model_state.bin'))
    model = model.to(device)

    test_acc, test_loss = eval_model(model, test_data_loader, loss_fn, device, len(test_texts))
    print(f'Test loss {test_loss} accuracy {test_acc}')


if __name__ == "__main__":
    main()
