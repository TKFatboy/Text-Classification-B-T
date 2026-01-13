import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp import word_vector


print("Loading Thai2Vec model...")
wv_wrapper = word_vector.WordVector(model_name="thai2fit_wv")
wv = wv_wrapper.get_model() 

file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)
df_small = df.head(5).copy() # ลองแค่ 5 ข่าวเหมือนเดิม

# Stopwords
stop_words = set(thai_stopwords())
my_custom_stops = {' ', '\n', '\t', '“', '”', '(', ')', '[', ']', '-', '.', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
stop_words.update(my_custom_stops)

# --- ฟังก์ชันเปลี่ยนประโยคให้เป็น Vector (ด้วยการหาค่าเฉลี่ย) ---
def get_sentence_vector(text):
    tokens = word_tokenize(str(text), engine='newmm')
    
    # กรองคำ และดึง Vector ของแต่ละคำมาเก็บไว้ใน list
    vecs = []
    for word in tokens:
        if word not in stop_words and word.strip() != '':
            try:
                # แปลงคำเป็น Vector (ขนาด 300 หรือ 400 มิติ แล้วแต่โมเดล)
                vec = wv.get_vector(word)
                vecs.append(vec)
            except:
                # ถ้าคำนี้ไม่มีในโมเดล Thai2Vec (เช่น ชื่อคนแปลกๆ) ก็ข้ามไป
                pass
    
    # ถ้าไม่มีคำไหนแปลงได้เลย (เช่น ข่าวว่างเปล่า) ให้คืนค่า 0
    if len(vecs) == 0:
        return np.zeros(wv.get_model().vector_size)
    
    # เอา Vector ของทุกคำมาหาค่าเฉลี่ย (Mean)
    sentence_vector = np.mean(vecs, axis=0)
    return sentence_vector

print("Converting text to Thai2Vec vectors...")
# สร้างคอลัมน์ใหม่ เก็บ Vector
# ผลลัพธ์จะเป็น Array ยาวๆ ในแต่ละช่อง
df_small['vector'] = df_small['body_text'].apply(get_sentence_vector)

print("\n--- ตัวอย่าง Vector ของข่าวแรก ---")
print(f"ขนาด Vector: {df_small['vector'].iloc[0].shape}") # เช็คขนาด (ปกติจะเป็น 300 หรือ 400)
print(df_small['vector'].iloc[0][:20]) # ดูตัวเลข 20 ตัวแรก

# หมายเหตุ: ข้อมูลตอนนี้พร้อมส่งเข้า model.fit() แล้วครับ
# แต่ต้องระวังนิดนึงตอนส่งเข้า sklearn ต้องแปลง list ของ array ให้เป็น matrix ใหญ่
X = np.vstack(df_small['vector'].values)
print(f"\nขนาด Matrix ที่จะส่งเข้า Model: {X.shape}")