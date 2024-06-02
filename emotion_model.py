import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

csv_file_path = '/content/tweet_emotions.csv'
text_file_path = '/content/test.txt'
stopwords_file_path = '/content/stopwords.txt'

lyrics_df_csv = pd.read_csv(csv_file_path)

with open(text_file_path, 'r', encoding='utf-8') as file:
    text_data = file.readlines()

text_df = pd.DataFrame([line.split(';') for line in text_data], columns=['content', 'sentiment'])

lyrics_df = pd.concat([lyrics_df_csv, text_df], ignore_index=True)

with open(stopwords_file_path, 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

def remove_special_chars(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

def remove_emails(text):
    return re.sub(r'\S+@\S+', '', text)

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def get_upper_idx(lyrics):
    upper_idx = []
    idx = 0
    for sentence in lyrics:
        sentence = remove_special_chars(sentence)
        try:
            if sentence[0].isupper():
                upper_idx.append(idx)
            idx = idx + 1
        except:
            pass
    return upper_idx

def join_sentence(index_list, corpus):
    result = []
    for i in range(len(index_list)):
        try:
            result.append(" ".join(corpus[index_list[i]:index_list[i+1]]))
        except IndexError:
            result.append(corpus[index_list[-1] - 1])
    return result

def lyrics_to_corpus(lyrics):
    lyrics = re.split(r'\r\n|\n', lyrics)
    corpus = join_sentence(get_upper_idx(lyrics), lyrics)
    corpus = list(set(corpus))
    return corpus

def sentence_preprocessing(corpus, stopwords):
    corpus = [sentence.strip().lower() for sentence in corpus]
    corpus = [remove_emails(sentence) for sentence in corpus]
    corpus = [remove_html_tags(sentence) for sentence in corpus]
    corpus = [remove_special_chars(sentence) for sentence in corpus]
    corpus = [" ".join([word for word in sentence.split() if word not in stopwords]) for sentence in corpus]
    return corpus

lyrics_df['processed_lyrics'] = lyrics_df['content'].apply(lambda x: " ".join(sentence_preprocessing(lyrics_to_corpus(x), stopwords)))

filtered_df = lyrics_df[lyrics_df['sentiment'].isin(['love', 'happy', 'sadness'])].copy()

label_encoder = LabelEncoder()
filtered_df.loc[:, 'sentiment'] = label_encoder.fit_transform(filtered_df['sentiment'])

train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings = tokenizer(train_df['processed_lyrics'].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df['processed_lyrics'].tolist(), truncation=True, padding=True, max_length=512)

train_dataset = EmotionDataset(train_encodings, train_df['sentiment'].tolist())
test_dataset = EmotionDataset(test_encodings, test_df['sentiment'].tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} loss: {loss.item()}')

model_save_path = '/content/emotion_analysis_model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")