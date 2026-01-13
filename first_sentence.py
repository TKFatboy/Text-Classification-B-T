from pythainlp.tokenize import word_tokenize
import torch
import os

device = torch.device("cuda")
if torch.cuda.is_available():
    print(f"PyTorch is running on: {device}")
else:
    print("PyTorch is running on: CPU")

file_name = "prachatai(TK Edition).csv"

if not os.path.exists(file_name):
    print(f"Error: file not found '{file_name}'")
else:

    with open(file_name, 'r', encoding='utf-8') as f:
        # ใช้ .readline() เพื่ออ่านบรรทัดแรกเท่านั้น
        first_sentence = f.readline().strip() 

    if first_sentence:
        print(f"บรรทัดที่ 1 ที่ถูกดึงมา: {first_sentence}")
        
        tokens = word_tokenize(first_sentence, engine='newmm')
        
        print("\n--- ผลลัพธ์การแยกคำ ---")
        print(f"คำที่แยกได้: {tokens}")
        print(f"จำนวนคำ: {len(tokens)}")
    else:
        print(f"Error")