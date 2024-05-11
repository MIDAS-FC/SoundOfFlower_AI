import boto3
from io import BytesIO
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.nn.functional import softmax

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_file_path = 'kobert_emotion_model.pt'  
model_state_dict = torch.load(model_file_path, map_location=device)

# s3 = boto3.client('s3')
# bucket_name = 'soundofflower'  
# object_name = 'kobert_emotion_model.pt'  
# response = s3.get_object(Bucket=bucket_name, Key=object_name)
# model_state_dict = torch.load(BytesIO(response['Body'].read()), map_location=device)

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=6, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

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

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(model_state_dict)
model.eval()

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
                'attention_mask': encoding['attention_mask'].flatten()
            })
            self.labels.append(torch.tensor(int(i[label_idx])))

    def __getitem__(self, idx):
        return {
            'input_ids': self.sentences[idx]['input_ids'],
            'attention_mask': self.sentences[idx]['attention_mask'],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.sentences)

global emotion_counts
emotion_counts = np.zeros(6)  
emotions = ["분노", "기쁨", "불안", "당황", "슬픔", "중립"]

def predict(predict_sentence):
    global emotion_counts
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, 64, True, False)
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()
    for batch_data in test_dataloader:
        
        token_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)

        out = model(token_ids=token_ids, valid_length=torch.sum(token_ids != 0, dim=1))
        logits = out.detach().cpu().numpy()
        probabilities = softmax(torch.tensor(logits), dim=1).numpy() 
        emotion_counts += probabilities[0]  
        predicted_emotion = emotions[np.argmax(probabilities)]
        print(f">> 입력하신 내용에서 {predicted_emotion} 느껴집니다.")

    print("Current emotion ratio:", emotion_counts)  

if __name__ == "__main__":
    predict(input())

