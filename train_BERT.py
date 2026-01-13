import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# ==========================================
# 1. Setup & Data Loading
# ==========================================
print("--- Step 1: Setup & Data Loading ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)

label_cols = ['politics', 'human_rights', 'quality_of_life', 'international',
              'social', 'environment', 'economics', 'culture', 'labor',
              'national_security', 'ict', 'education']

# เตรียมข้อมูล (Multi-Label ของจริง)
texts = df['body_text'].values
labels = df[label_cols].values

# --- [ไฮไลท์สำคัญ: คำนวณน้ำหนักให้แต่ละหมวด] ---
# สูตร: (จำนวนข่าวทั้งหมด - จำนวนข่าวหมวดนั้น) / จำนวนข่าวหมวดนั้น
# ยิ่งข่าวน้อย น้ำหนักยิ่งเยอะ
num_samples = len(df)
counts = df[label_cols].sum().values
pos_weights = (num_samples - counts) / counts
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float).to(device)
print("\n--- Calculated Class Weights (น้ำหนักความสำคัญ) ---")

for i, col in enumerate(label_cols):
    print(f"{col}: {pos_weights[i]:.2f}")
# คุณจะเห็นว่า Politics น้ำหนักต่ำ (~2.0) แต่ ICT น้ำหนักสูง (~30.0)

# แบ่ง Train/Test
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# ==========================================
# 2. Custom Dataset
# ==========================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
       
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

# เพิ่ม Batch Size ได้นิดหน่อยถ้า GPU ไหว (16 หรือ 32)
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. Model & Training
# ==========================================
print(f"\n--- Step 2: Loading BERT Model ---")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5) # ขยับ LR ขึ้นนิดนึงเพราะข้อมูลเยอะ

# [ไฮไลท์สำคัญ: ใส่ Weight เข้าไปใน Loss Function]
# pos_weight จะบังคับให้โมเดลใส่ใจคลาสเล็กๆ มากขึ้น
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

print("\n--- Step 3: Start Fine-tuning (Weighted) ---")
EPOCHS = 5 # ลอง 5 รอบก่อน (ถ้ายังไม่แม่นค่อยเพิ่มเป็น 10)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # คำนวณ Loss แบบถ่วงน้ำหนัก
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 4. Interactive Prediction
# ==========================================
print("\n--- Step 4: Testing ---")
def predict_news(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits) # แปลงเป็น %

    probs = probs.cpu().detach().numpy()[0]
    print("-" * 50)
    print(f"[ข่าว]: {text[:80]}...")
    print(">> หมวดหมู่ที่ได้:")
    found = False
    # Threshold 0.5 (มาตรฐาน)
    for i, prob in enumerate(probs):
        if prob > 0.5:
            print(f"   ✅ {label_cols[i]}: {prob*100:.2f}%")
            found = True

    if not found:
        # ถ้าไม่มีใครผ่านเกณฑ์ ให้โชว์ Top 3 ที่เป็นไปได้สูงสุด
        print("   (ไม่ผ่านเกณฑ์ 50% แต่สูงสุดคือ):")
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"      - {label_cols[idx]}: {probs[idx]*100:.2f}%")

# ทดสอบชุดใหญ่
news_list = [
    "แฮคเกอร์โจมตีระบบธนาคาร ขโมยข้อมูลลูกค้าไปขายต่อ", # ควรออก ICT (+Politics/Econ)
    "ชาวบ้านชุมนุมคัดค้านเหมืองแร่ เรียกร้องให้ตรวจสอบผลกระทบสิ่งแวดล้อม", # Env + Human Rights
    "แรงงานเรียกร้องขึ้นค่าแรงขั้นต่ำ รัฐบาลรับปากจะพิจารณา", # Labor + Politics + Econ
    "ภาพยนตร์ไทยเรื่องใหม่กวาดรางวัลในเทศกาลหนังเมืองคานส์" # Culture
]
for news in news_list:
    predict_news(news)