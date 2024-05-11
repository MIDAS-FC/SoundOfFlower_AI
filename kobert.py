from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm  

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.utils

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

vocab_size = tokenizer.vocab_size

print("Vocab size:", vocab_size)

df = pd.read_excel('data/감성대화말뭉치(최종데이터)_Training.xlsx')
df.to_csv('data/감성대화말뭉치(최종데이터)_Training.csv')
df1 = pd.read_csv('data/감성대화말뭉치(최종데이터)_Training.csv')

df2 = df1[['감정_대분류', '사람문장1']]
df2.head(2)
df2['감정_대분류'].unique()
df2.loc[(df2['감정_대분류'] == "분노"), '감정_대분류'] = 0
df2.loc[(df2['감정_대분류'] == "기쁨"), '감정_대분류'] = 1
df2.loc[(df2['감정_대분류'] == "불안"), '감정_대분류'] = 2
df2.loc[(df2['감정_대분류'] == "당황"), '감정_대분류'] = 3
df2.loc[(df2['감정_대분류'] == "슬픔"), '감정_대분류'] = 4
df2.loc[(df2['감정_대분류'] == "상처"), '감정_대분류'] = 4
df2.loc[(df2['감정_대분류'] == "중립"), '감정_대분류'] = 5

data_list = []
for ques, label in zip (df2['사람문장'], df2['감정_대분류']):
    data = []
    data.append(ques)
    data.append(str(label))

    data_list.append(data)

print(data)
print(data_list[:10])

dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=32)

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len, pad=True, pair=False):
        self.sentences = []
        self.labels = []

        self.tokenizer = tokenizer
        self.max_len = max_len

        for i in dataset:
            encoding = self.tokenizer.encode_plus(
                text=i[sent_idx],
                add_special_tokens=True,  
                max_length=self.max_len,
                padding='max_length' if pad else None, 
                truncation=True, 
                return_attention_mask=True,  
                return_tensors='pt' 
            )
            self.sentences.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding else torch.zeros(self.max_len)
            })

            self.labels.append(torch.tensor(int(i[label_idx]), dtype=torch.long))

    def __getitem__(self, idx):
    
        return {
            'input_ids': self.sentences[idx]['input_ids'],
            'attention_mask': self.sentences[idx]['attention_mask'],
            'token_type_ids': self.sentences[idx]['token_type_ids'],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

train_dataset = BERTDataset(dataset_train, 0, 1, tokenizer, max_len)
test_dataset = BERTDataset(dataset_test, 0, 1, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes = 6,
                dr_rate = None,
                params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate = 0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_loader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()  
    for batch_id, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {e+1} Training")):
        optimizer.zero_grad()
        token_ids = batch_data['input_ids'].to(device)
        label = batch_data['label'].to(device)
        valid_length = torch.sum(token_ids != 0, axis=1)  

        
        out = model(token_ids=token_ids, valid_length=valid_length)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print(f"epoch {e+1} batch id {batch_id+1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id+1)}")
    print(f"epoch {e+1} train acc {train_acc / (batch_id+1)}")

    model.eval() 
    for batch_id, batch_data in enumerate(tqdm(test_loader, desc=f"Epoch {e+1} Testing")):
        token_ids = batch_data['input_ids'].to(device)
        label = batch_data['label'].to(device)
        valid_length = torch.sum(token_ids != 0, axis=1) 
        out = model(token_ids=token_ids, valid_length=valid_length)
        test_acc += calc_accuracy(out, label)
    print(f"epoch {e+1} test acc {test_acc / (batch_id+1)}")


# 모든 에폭이 끝난 후 모델 저장
torch.save(model.state_dict(), 'kobert_emotion_model.pt')
print("모델 저장 완료! 파일명: 'kobert_emotion_model.pt'")

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()

    for batch_data in test_dataloader:
        token_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        label = batch_data['label'].to(device)

        out = model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))

        test_eval = []
        logits = out.detach().cpu().numpy()
        for i in logits:
            if np.argmax(i) == 0:
                test_eval.append("분노가")
            elif np.argmax(i) == 1:
                test_eval.append("기쁨이")
            elif np.argmax(i) == 2:
                test_eval.append("불안이")
            elif np.argmax(i) == 3:
                test_eval.append("당황이")
            elif np.argmax(i) == 4:
                test_eval.append("슬픔이")
            elif np.argmax(i) == 5:
                test_eval.append("중립이")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

def main():
    end = 1
    while end == 1:
        sentence = input("하고 싶은 말을 입력해주세요 (종료하려면 '0'을 입력) : ")
        if sentence == "0":
            break
        predict(sentence)
        print("\n")

if __name__ == "__main__":
    main()
