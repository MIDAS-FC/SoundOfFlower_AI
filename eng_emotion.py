df = pd.read_excel('finalEngData.xlsx')
df.to_csv('finalEngData.csv')
df1 = pd.read_csv('finalEngData.csv')

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np

texts = df1['사람문장'].tolist()
labels = df1['감정'].tolist()
emotion_map = {
    "happiness": 0,
    "love": 1,
    "sadness": 2
}
df1['label'] = df1['감정'].map(emotion_map)

texts = df1['사람문장'].tolist()
labels = df1['label'].tolist()  

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

train_texts = [str(text) for text in train_texts]
val_texts = [str(text) for text in val_texts]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # 숫자 라벨 사용
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df1['감정'].unique()))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./eng_emotion_model")
tokenizer.save_pretrained("./eng_emotion_model")