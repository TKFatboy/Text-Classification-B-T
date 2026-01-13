import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
# [แก้ไขใหม่ 1] เพิ่ม import gensim
from gensim.models import Word2Vec 

# ==========================================
# 1. เตรียมข้อมูล (Data Preparation)
# ==========================================
print("--- Step 1: Loading Data & Custom Word2Vec ---")

# [แก้ไขใหม่ 2] โหลดโมเดลที่เราสร้างเอง
print("Loading 'custom_word2vec.model'...")
try:
    # โหลดไฟล์ที่เราเพิ่ง save ไปเมื่อกี้
    w2v_model = Word2Vec.load("custom_word2vec.model") 
    print("-> โหลด Custom Word2Vec สำเร็จ!")
except:
    print("Error: หาไฟล์ custom_word2vec.model ไม่เจอ! (กรุณารัน create_word2vec.py ก่อน)")
    exit()

# 1.2 โหลดไฟล์ CSV
file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)

# 1.3 จัดการ Label (เหมือนเดิม)
label_cols = ['politics', 'human_rights', 'quality_of_life', 'international', 
              'social', 'environment', 'economics', 'culture', 'labor', 
              'national_security', 'ict', 'education']

def get_category(row):
    for col in label_cols:
        if row[col] == 1:
            return col
    return 'other'

df['category_name'] = df.apply(get_category, axis=1)
df = df[df['category_name'] != 'other'].copy()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['category_name'])
num_classes = len(encoder.classes_)
print(f"จำนวนหมวดหมู่ข่าว (Classes): {num_classes}")

# 1.4 ฟังก์ชันแปลง Text -> Average Vector (แก้ให้ใช้ Gensim)
stop_words = set(thai_stopwords())

def get_avg_vector(text):
    tokens = word_tokenize(str(text), engine='newmm')
    vecs = []
    for word in tokens:
        if word not in stop_words and word.strip() != '':
            # [แก้ไขใหม่ 3] เช็คคำใน Gensim Model
            if word in w2v_model.wv.key_to_index:
                vec = w2v_model.wv[word] # ดึง vector ออกมา
                vecs.append(vec)
    
    if len(vecs) == 0:
        return np.zeros(300) 
    
    return np.mean(vecs, axis=0)

print("Converting text to vectors... (อาจใช้เวลาสักครู่)")
X_numpy = np.vstack(df['body_text'].apply(get_avg_vector).values)
y_numpy = y_encoded

# 1.5 แปลง Numpy -> PyTorch Tensor
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
y_tensor = torch.tensor(y_numpy, dtype=torch.long)

# แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# สร้าง DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 2. สร้างโมเดล (Simple MLP)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# ตั้งค่า Model
input_dim = 300   # ต้องเท่ากับขนาด vector ของ custom model (300)
hidden_dim = 256  
output_dim = num_classes

model = SimpleMLP(input_dim, hidden_dim, output_dim)
print("\n--- Model Architecture ---")
print(model)

# ==========================================
# 3. กำหนด Loss & Optimizer
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. เริ่มสอนโมเดล (Training Loop)
# ==========================================
print("\n--- Step 2: Training Start ---")
num_epochs = 100 

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0: # ปรินท์ทุก 10 รอบ
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 5. วัดผล (Evaluation)
# ==========================================
print("\n--- Step 3: Evaluation ---")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    print("\n--- ตัวอย่างการทายผล ---")
    print(f"หมวดจริง: {encoder.inverse_transform([y_test[0].item()])[0]}")
    print(f"โมเดลทาย: {encoder.inverse_transform([predicted[0].item()])[0]}")