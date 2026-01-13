import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW


# ==========================================
# 1. การตั้งค่าเบื้องต้น (Setup)
# ==========================================
print("--- Step 1: Setup ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
CACHE_FILE = "cached_data_bert.pt" # ไฟล์สำหรับเก็บข้อมูลที่แปลงแล้ว

file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)

label_cols = ['politics', 'human_rights', 'quality_of_life', 'international', 
              'social', 'environment', 'economics', 'culture', 'labor', 
              'national_security', 'ict', 'education']

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูล (Data Preparation)
# ==========================================
def prepare_data(df, tokenizer, max_len=256):
    # กรณีที่ 1: เจอไฟล์ Cache -> โหลดเลย เร็วมาก
    if os.path.exists(CACHE_FILE):
        print(f"✅ เจอไฟล์ Cache '{CACHE_FILE}'... กำลังโหลดข้อมูล (สูตรน้ำหนักต้องเป็นตัวใหม่นะ!)")
        saved_data = torch.load(CACHE_FILE)
        
        train_dataset = TensorDataset(saved_data['train_inputs'], saved_data['train_masks'], saved_data['train_labels'])
        test_dataset = TensorDataset(saved_data['test_inputs'], saved_data['test_masks'], saved_data['test_labels'])
        pos_weights_tensor = saved_data['pos_weights'].to(device)
        
        print("-> โหลดข้อมูลเสร็จสิ้น!")
        return train_dataset, test_dataset, pos_weights_tensor

    # กรณีที่ 2: ไม่เจอไฟล์ -> ทำใหม่ (คำนวณน้ำหนักสูตรใหม่)
    print(f"⚠️ ไม่เจอไฟล์ Cache... กำลังเริ่มกระบวนการแปลงข้อมูลใหม่")
    
    texts = df['body_text'].values
    labels = df[label_cols].values 

    # --- [แก้สูตรน้ำหนักตรงนี้: ใช้ Sqrt ลดความแรง] ---
    num_samples = len(df)
    counts = df[label_cols].sum().values
    
    # สูตรเดิม: (Total - Count) / Count  <-- แรงไป
    # สูตรใหม่: Sqrt( (Total - Count) / Count ) <-- นุ่มนวลขึ้น
    raw_weights = (num_samples - counts) / np.maximum(counts, 1) # กันหาร 0
    pos_weights = np.sqrt(raw_weights) 
    
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float).to(device)
    
    print("\n--- Calculated Class Weights (Sqrt Dampened) ---")
    for i, col in enumerate(label_cols):
        print(f"  - {col}: {pos_weights[i]:.2f}")

    # แบ่ง Train/Test
    train_texts, test_texts, train_y, test_y = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Batch Tokenization
    def batch_encode(text_list):
        return tokenizer.batch_encode_plus(
            list(text_list),
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

    print("   -> Tokenizing Train Data...")
    train_enc = batch_encode(train_texts)
    print("   -> Tokenizing Test Data...")
    test_enc = batch_encode(test_texts)

    # แปลง Label เป็น Tensor
    train_labels = torch.tensor(train_y, dtype=torch.float)
    test_labels = torch.tensor(test_y, dtype=torch.float)

    # บันทึก Cache
    print(f"💾 กำลังบันทึก Cache ลงไฟล์ '{CACHE_FILE}'...")
    torch.save({
        'train_inputs': train_enc['input_ids'],
        'train_masks': train_enc['attention_mask'],
        'train_labels': train_labels,
        'test_inputs': test_enc['input_ids'],
        'test_masks': test_enc['attention_mask'],
        'test_labels': test_labels,
        'pos_weights': pos_weights_tensor
    }, CACHE_FILE)
    
    train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_enc['input_ids'], test_enc['attention_mask'], test_labels)
    
    return train_dataset, test_dataset, pos_weights_tensor

# เรียกใช้ฟังก์ชัน
train_dataset, test_dataset, pos_weights_tensor = prepare_data(df, tokenizer)

# สร้าง DataLoader
BATCH_SIZE = 16 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==========================================
# 3. สร้างและเทรนโมเดล (Model & Training)
# ==========================================
print(f"\n--- Step 2: Loading BERT Model ---")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

# Loss Function แบบมีน้ำหนัก
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

print("\n--- Step 3: Start Fine-tuning ---")
EPOCHS = 5 # รอบเดียวพอ!

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # ลำดับข้อมูลใน TensorDataset: 0=input_ids, 1=mask, 2=labels
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ==========================================
# 4. บันทึกโมเดล (Save Model)
# ==========================================
print("\n--- Saving Model ---")
output_dir = "./my_bert_multilabel_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ บันทึกโมเดลเสร็จสิ้นที่: {output_dir}")

# ==========================================
# 5. ทดสอบทำนายผล (Prediction)
# ==========================================
print("\n--- Step 4: Testing & Prediction ---")

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
        probs = torch.sigmoid(outputs.logits) 
        
    probs = probs.cpu().detach().numpy()[0]
    
    print("-" * 50)
    print(f"[ข่าว]: {text[:80]}...")
    print(">> หมวดหมู่ที่ได้:")
    
    found = False
    # Threshold 0.5
    for i, prob in enumerate(probs):
        if prob > 0.5: 
            print(f"   ✅ {label_cols[i]}: {prob*100:.2f}%")
            found = True
            
    if not found:
        print("   (คะแนนไม่ถึง 50% แต่สูงสุดคือ):")
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"      - {label_cols[idx]}: {probs[idx]*100:.2f}%")

# ชุดทดสอบ
news_list = [
    "แฮคเกอร์โจมตีระบบธนาคาร ขโมยข้อมูลลูกค้าไปขายต่อ", 
    "ชาวบ้านชุมนุมคัดค้านเหมืองแร่ เรียกร้องให้ตรวจสอบผลกระทบสิ่งแวดล้อม", 
    "แรงงานเรียกร้องขึ้นค่าแรงขั้นต่ำ รัฐบาลรับปากจะพิจารณา", 
    "ภาพยนตร์ไทยเรื่องใหม่กวาดรางวัลในเทศกาลหนังเมืองคานส์" 
]

for news in news_list:
    predict_news(news)