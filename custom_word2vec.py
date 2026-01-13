import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from gensim.models import Word2Vec
import logging

# ตั้งค่าให้โชว์ log ว่าเทรนไปถึงไหนแล้ว
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Loading Data...")
file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)

print("Tokenizing & Cleaning...")
stop_words = set(thai_stopwords())
my_custom_stops = {' ', '\n', '\t', '“', '”', '(', ')', '[', ']', '-', '.', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
stop_words.update(my_custom_stops)

# เตรียมประโยคสำหรับสอน
sentences = []
for text in df['body_text']:
    # ตัดคำ
    tokens = word_tokenize(str(text), engine='newmm')
    # กรองคำ
    clean_tokens = [w for w in tokens if w not in stop_words and w.strip() != '']
    if len(clean_tokens) > 0:
        sentences.append(clean_tokens)

print(f"ได้ประโยคสำหรับสอน: {len(sentences)} ประโยค")

print("Training Word2Vec...")
# vector_size=300: ขนาดเท่าตัวเก่า
# window=5: ดูบริบท 5 คำหน้าหลัง
# min_count=1: **สำคัญมาก** คำไหนโผล่มาแค่ครั้งเดียวก็เอาหมด (ไม่ทิ้ง Anonymous)
# epochs=50: วนอ่านข่าวเดิม 50 รอบให้จำแม่นๆ
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4, epochs=50)

# บันทึกโมเดล
model.save("custom_word2vec.model")
print("\n[SUCCESS] สร้างโมเดลเสร็จแล้ว! บันทึกเป็นไฟล์ 'custom_word2vec.model'")

# ทดสอบของหน่อย
print("\n--- ทดสอบคำศัพท์ ---")
check_words = ['Anonymous', 'แฮคเกอร์', 'ประยุทธ์', 'เลือกตั้ง']
for w in check_words:
    if w in model.wv:
        print(f"คำว่า '{w}' -> เจอ! (ใกล้เคียงกับ: {model.wv.most_similar(w, topn=3)})")
    else:
        print(f"คำว่า '{w}' -> ไม่เจอ")