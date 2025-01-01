# 引入所需的库
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib
# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载 SQuAD 数据集
dataset = load_dataset('squad')

# 打印数据集的基本信息
print(f"Training examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")

# 训练数据集示例
train_data = dataset['train']
val_data = dataset['validation']

# 数据集类定义，适应SQuAD数据集
class QADataset(Dataset):
    def __init__(self, data, tokenizer, source_len, target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取 SQuAD 数据中的 question 和 context
        question = self.data[index]['question']
        context = self.data[index]['context']
        answer = self.data[index]['answers']['text'][0]  # 假设只取第一个答案

        # 对输入的 question 和 context 进行编码
        source = self.tokenizer.encode_plus(
            question, context,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 对目标答案进行编码
        target = self.tokenizer.encode(
            answer,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': source['input_ids'].squeeze(0).to(device),
            'attention_mask': source['attention_mask'].squeeze(0).to(device),
            'labels': target.squeeze(0).to(device)
        }

# 初始化 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# 设置数据集的最大长度
source_len = 512
target_len = 128

# 创建数据集实例
train_dataset = QADataset(train_data, tokenizer, source_len, target_len)
val_dataset = QADataset(val_data, tokenizer, source_len, target_len)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 定义优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

# 训练和评估函数
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}], Loss: {loss.item():.4f}")

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

# 开始训练
epochs = 2
for epoch in range(epochs):
    train(model, train_loader, optimizer, epoch)
    evaluate(model, val_loader)

# 保存模型
model.save_pretrained('./qa_model')
tokenizer.save_pretrained('./qa_model')

#plotting the loss
# import matplotlib.pyplot as plt
# plt.plot(avg_loss)
# plt.title('Training loss')
# plt.xlabel('Epochs')
