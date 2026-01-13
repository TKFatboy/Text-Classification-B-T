import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

file_name = r"d:\year4\สหกิจ\prachatai_test.csv"
df = pd.read_csv(file_name)

# เลือกมาแค่ 5 ข่าวแรกเพื่อทดสอบ
df_small = df.head(5).copy()

stop_words = set(thai_stopwords())
my_custom_stops = {' ', '\n', '\t', '“', '”', '(', ')', '[', ']', '-', '.', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} # เพิ่มตัวเลขเข้าไปด้วย
stop_words.update(my_custom_stops)

def text_process(text):
    # ตัดคำ
    tokens = word_tokenize(text, engine='newmm')
    # กรองคำ
    clean_tokens = []
    for w in tokens:
        if w not in stop_words and w.strip() != '':
            clean_tokens.append(w)
    return ' '.join(clean_tokens)    # รวมกลับเป็นสตริงคั่นด้วยช่องว่าง (เพราะ TfidfVectorizer ชอบ input แบบ string ยาวๆ ที่มีเว้นวรรค)

# ใช้ฟังก์ชันกับข้อมูล 5 แถวแรก
print("กำลังทำความสะอาดข้อมูล...")
df_small['clean_text'] = df_small['body_text'].apply(text_process)

print("\n--- ตัวอย่างข้อความหลัง Clean ---")
print(df_small['clean_text'].iloc[0][:100] + "...") # ดูข่าวแรก

# Vectorization ด้วย TF-IDF
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), token_pattern=None) # สั่งให้มันใช้ช่องว่างที่เราทำไว้แยกคำ
tfidf_matrix = tfidf_vectorizer.fit_transform(df_small['clean_text'])

print("\n--- ผลลัพธ์ Feature Extraction ---")
print(f"ขนาดของ Matrix (จำนวนข่าว, จำนวนคำศัพท์ทั้งหมด): {tfidf_matrix.shape}")

# ลองดูคะแนนของคำในข่าวแรก
feature_names = tfidf_vectorizer.get_feature_names_out()
first_document_vector = tfidf_matrix[0]

# แสดงคำที่มีคะแนนสูงสุด 5 อันดับแรกในข่าวที่ 1
print("\n--- คำเด่นประจำข่าวที่ 1 (Top Keywords by TF-IDF) ---")
df_tfidf = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
print(df_tfidf.sort_values(by=["tfidf"], ascending=False).head(10))