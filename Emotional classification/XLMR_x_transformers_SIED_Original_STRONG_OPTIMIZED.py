# !pip install -q transformers sentencepiece x-transformers optuna scikit-learn
# !apt-get -qq update
# !apt-get -qq install -y unrar


# %%

import os
import json
import random
import warnings
import subprocess
import shutil
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from x_transformers import Decoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    hamming_loss,
    multilabel_confusion_matrix
)

import optuna


# %%

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", device)

# %%

MODEL_NAME = "xlm-roberta-base"

MAX_LEN = 128
EPOCHS = 100
BATCH_SIZE = 64
NUM_LABELS = 6

N_TRIALS = 25
OPTUNA_EPOCHS = 35

LABEL_COLUMNS = [
    "Happiness",
    "Sadness",
    "Anger",
    "Disgust",
    "Surprise",
    "Fear"
]

TEXT_COLUMN = "Tweets"


# %%

# ถ้าใช้ไฟล์ .rar ที่อัปโหลดมาโดยตรงใน Google Colab
RAR_PATH = "/content/SIED-Thai.rar"
CSV_PATH = "/content/SIED-Thai.csv"
EXTRACT_ROOT = "/content/sied_thai_extracted"

if not os.path.exists(CSV_PATH) and os.path.exists(RAR_PATH):
    os.makedirs(EXTRACT_ROOT, exist_ok=True)

    if shutil.which("unrar") is None:
        raise RuntimeError(
            "ไม่พบคำสั่ง unrar กรุณารัน cell ติดตั้งด้านบนก่อน "
            "หรือ extract SIED-Thai.rar แล้วอัปโหลด SIED-Thai.csv"
        )

    subprocess.run(
        ["unrar", "x", "-o+", RAR_PATH, EXTRACT_ROOT + "/"],
        check=True
    )

    CSV_PATH = os.path.join(EXTRACT_ROOT, "SIED-Thai.csv")

# fallback สำหรับ local/sandbox
if not os.path.exists(CSV_PATH):
    CSV_PATH = "/mnt/data/SIED-Thai.csv"

print("SIED CSV:", CSV_PATH)


df = pd.read_csv(CSV_PATH)

required_columns = [TEXT_COLUMN] + LABEL_COLUMNS

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"dataset ไม่มี column: {col}")

df = df[required_columns].copy()
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()

for col in LABEL_COLUMNS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0).astype(int)

print("TOTAL ROWS:", len(df))
print("\nORIGINAL LABEL DISTRIBUTION")
print(df[LABEL_COLUMNS].sum())

print("\nORIGINAL LABEL CARDINALITY")
print(df[LABEL_COLUMNS].sum(axis=1).value_counts().sort_index())


# %%

# Original SIED-Thai ไม่มี train/validation/test แยกมาให้
# ใช้ split 70/15/15 เหมือน Strong Optimized framework
# และใช้ stratify ตาม label combination เพื่อลดความคลาดเคลื่อนของ class distribution

def make_stratify_key(frame):
    label_part = frame[LABEL_COLUMNS].astype(str).agg("".join, axis=1)
    card_part = frame[LABEL_COLUMNS].sum(axis=1).astype(str)
    key = card_part + "_" + label_part

    counts = key.value_counts()
    key = key.where(key.map(counts) >= 2, "rare")

    return key


stratify_key = make_stratify_key(df)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True,
    stratify=stratify_key
)

temp_stratify_key = make_stratify_key(temp_df)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
    stratify=temp_stratify_key
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("TRAIN:", len(train_df))
print("VALIDATION:", len(val_df))
print("TEST:", len(test_df))

print("\nTRAIN LABEL DISTRIBUTION")
print(train_df[LABEL_COLUMNS].sum())

print("\nVAL LABEL DISTRIBUTION")
print(val_df[LABEL_COLUMNS].sum())

print("\nTEST LABEL DISTRIBUTION")
print(test_df[LABEL_COLUMNS].sum())

print("\nTRAIN LABEL CARDINALITY")
print(train_df[LABEL_COLUMNS].sum(axis=1).value_counts().sort_index())

print("\nVAL LABEL CARDINALITY")
print(val_df[LABEL_COLUMNS].sum(axis=1).value_counts().sort_index())

print("\nTEST LABEL CARDINALITY")
print(test_df[LABEL_COLUMNS].sum(axis=1).value_counts().sort_index())


# %%

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%

class EmotionDataset(Dataset):

    def __init__(self, df):

        self.texts = df[TEXT_COLUMN].tolist()

        self.labels = df[LABEL_COLUMNS].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        labels = self.labels[idx]

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }
train_dataset = EmotionDataset(train_df)

val_dataset = EmotionDataset(val_df)

test_dataset = EmotionDataset(test_df)

# %%

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_eval_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# %%

# pos_weight เดิมทำให้ recall สูง แต่ precision ตกได้
# จึงไม่ fix pos_weight ตายตัว แต่ให้ Optuna tune pos_weight_scale
# scale = 0.0 คือไม่ใช้ pos_weight
# scale = 1.0 คือใช้ pos_weight เต็มสูตร negative / positive

label_counts = train_df[LABEL_COLUMNS].sum().values.astype(np.float32)
total_samples = len(train_df)
neg_counts = total_samples - label_counts

pos_weights_raw = neg_counts / (label_counts + 1e-6)

pos_weights_raw = torch.tensor(
    pos_weights_raw,
    dtype=torch.float
).to(device)

def build_pos_weight(pos_weight_scale):
    pos_weight_scale = float(pos_weight_scale)

    weights = 1.0 + (pos_weights_raw - 1.0) * pos_weight_scale

    return weights.to(device)

print("LABEL COUNTS:", label_counts)
print("RAW POS WEIGHTS:", pos_weights_raw)
print("NO POS WEIGHT:", build_pos_weight(0.0))
print("HALF POS WEIGHT:", build_pos_weight(0.5))


# %%

class AttentionPooling(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, mask):

        scores = self.attention(x).squeeze(-1)

        scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return pooled

# %%

class EmotionModel(nn.Module):

    def __init__(
        self,
        depth=1,
        heads=2,
        attn_dropout=0.15,
        ff_dropout=0.15,
        ff_mult=2
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        # Fixed Thai encoder: train เฉพาะ decoder + pooling + classifier
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.decoder = Decoder(
            dim=hidden_size,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult
        )

        self.pooling = AttentionPooling(hidden_size)

        self.dropout = nn.Dropout(0.30)

        self.fc = nn.Linear(
            hidden_size,
            NUM_LABELS
        )

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        x = outputs.last_hidden_state

        x = self.decoder(x)

        x = self.pooling(x, attention_mask)

        x = self.dropout(x)

        logits = self.fc(x)

        return logits


# %%

def optimize_thresholds(
    y_true,
    y_probs,
    min_thr=0.20,
    max_thr=0.80,
    step=0.005
):

    thresholds = np.arange(
        min_thr,
        max_thr + 1e-9,
        step
    )

    best_thresholds = []

    for i in range(NUM_LABELS):

        best_thr = 0.5
        best_score = -1
        best_precision = -1

        true_support = y_true[:, i].sum()

        for thr in thresholds:

            preds = (y_probs[:, i] >= thr).astype(int)

            score = f1_score(
                y_true[:, i],
                preds,
                zero_division=0
            )

            precision = precision_score(
                y_true[:, i],
                preds,
                zero_division=0
            )

            # ถ้า F1 เท่ากัน ให้เลือก threshold ที่ precision สูงกว่า
            # ช่วยแก้ปัญหา optimized เดิมที่ recall สูงแต่ precision ต่ำ
            if (score > best_score) or (
                np.isclose(score, best_score) and precision > best_precision
            ):

                best_score = score
                best_precision = precision
                best_thr = thr

        best_thresholds.append(best_thr)

    return np.array(best_thresholds)


# %%

def evaluate(model, loader, thresholds=None):

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                attention_mask
            )

            probs = torch.sigmoid(logits)

            all_probs.append(
                probs.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    if thresholds is None:
        thresholds = np.array([0.5] * NUM_LABELS)

    preds = (all_probs >= thresholds).astype(int)

    micro_f1 = f1_score(
        all_labels,
        preds,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        preds,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        preds,
        average="weighted",
        zero_division=0
    )

    micro_precision = precision_score(
        all_labels,
        preds,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        all_labels,
        preds,
        average="macro",
        zero_division=0
    )

    micro_recall = recall_score(
        all_labels,
        preds,
        average="micro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        preds,
        average="macro",
        zero_division=0
    )

    per_label_f1 = f1_score(
        all_labels,
        preds,
        average=None,
        zero_division=0
    )

    per_label_precision = precision_score(
        all_labels,
        preds,
        average=None,
        zero_division=0
    )

    per_label_recall = recall_score(
        all_labels,
        preds,
        average=None,
        zero_division=0
    )

    subset_accuracy = accuracy_score(
        all_labels,
        preds
    )

    h_loss = hamming_loss(
        all_labels,
        preds
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "per_label_f1": per_label_f1,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "min_per_label_f1": float(np.min(per_label_f1)),
        "accuracy": subset_accuracy,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": h_loss,
        "labels": all_labels,
        "preds": preds,
        "probs": all_probs
    }


def objective_score(result):
    # objective ใหม่:
    # - macro_f1 สำคัญที่สุด เพราะต้องชนะราย class
    # - micro_f1 ยังสำคัญ
    # - min_per_label_f1 กันไม่ให้ class ใด class หนึ่งต่ำ โดยเฉพาะ Disgust
    # - subset accuracy ช่วยให้ทายชุด label ตรงขึ้น
    return (
        0.45 * result["macro_f1"]
        + 0.25 * result["micro_f1"]
        + 0.20 * result["min_per_label_f1"]
        + 0.10 * result["subset_accuracy"]
    )


# %%

def train_model(
    model,
    train_loader,
    val_loader,
    lr,
    weight_decay,
    epochs=None,
    pos_weight_scale=0.0,
    label_smoothing=0.0,
    loss_name="bce",
    focal_gamma=2.0,
    verbose=True
):

    if epochs is None:
        epochs = EPOCHS

    pos_weight = build_pos_weight(pos_weight_scale)

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        loop = tqdm(
            train_loader,
            disable=not verbose
        )

        for batch in loop:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if label_smoothing > 0:
                # smooth toward 0.5 for multi-label BCE
                labels_for_loss = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                labels_for_loss = labels

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):

                logits = model(
                    input_ids,
                    attention_mask
                )

                loss_matrix = bce_loss(
                    logits,
                    labels_for_loss
                )

                if loss_name == "focal":
                    probs = torch.sigmoid(logits)
                    pt = probs * labels + (1 - probs) * (1 - labels)
                    focal_weight = (1 - pt).pow(focal_gamma)
                    loss = (focal_weight * loss_matrix).mean()
                else:
                    loss = loss_matrix.mean()

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if verbose:
                loop.set_description(
                    f"Epoch {epoch+1}/{epochs}"
                )

                loop.set_postfix(
                    loss=loss.item()
                )

        if verbose:

            val_raw = evaluate(
                model,
                val_loader
            )

            val_thresholds = optimize_thresholds(
                val_raw["labels"],
                val_raw["probs"]
            )

            val_result = evaluate(
                model,
                val_loader,
                val_thresholds
            )

            print("\n======================")
            print(f"Epoch {epoch+1}")
            print("======================")

            print(
                f"VAL SUBSET ACC : {val_result['subset_accuracy']:.4f}"
            )

            print(
                f"VAL MICRO P/R/F1: "
                f"{val_result['micro_precision']:.4f} / "
                f"{val_result['micro_recall']:.4f} / "
                f"{val_result['micro_f1']:.4f}"
            )

            print(
                f"VAL MACRO P/R/F1: "
                f"{val_result['macro_precision']:.4f} / "
                f"{val_result['macro_recall']:.4f} / "
                f"{val_result['macro_f1']:.4f}"
            )

            print(
                f"VAL MIN LABEL F1: {val_result['min_per_label_f1']:.4f}"
            )

            print(
                f"VAL WEIGHTED F1: {val_result['weighted_f1']:.4f}"
            )

            print(
                f"VAL HAMMING LOSS: {val_result['hamming_loss']:.4f}"
            )

    return model


# %%

def objective(trial):

    # baseline trial จะถูก enqueue ไว้ก่อนหน้า
    depth = trial.suggest_int(
        "depth",
        1,
        3
    )

    heads = trial.suggest_categorical(
        "heads",
        [2, 4, 8]
    )

    # ลด dropout range จากเดิมที่สูงเกินไป เพราะเดิม optimized มีแนวโน้ม underfit / recall เยอะ precision ตก
    attn_dropout = trial.suggest_float(
        "attn_dropout",
        0.05,
        0.30
    )

    ff_dropout = trial.suggest_float(
        "ff_dropout",
        0.05,
        0.30
    )

    ff_mult = trial.suggest_int(
        "ff_mult",
        2,
        4
    )

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-4,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        5e-3,
        log=True
    )

    # จุดสำคัญ: ให้ Optuna เลือกว่าจะใช้ pos_weight แค่ไหน
    # ถ้า scale สูงเกินจะ recall สูงแต่ precision ลด
    pos_weight_scale = trial.suggest_float(
        "pos_weight_scale",
        0.0,
        0.60
    )

    label_smoothing = trial.suggest_float(
        "label_smoothing",
        0.0,
        0.04
    )

    loss_name = trial.suggest_categorical(
        "loss_name",
        ["bce", "focal"]
    )

    focal_gamma = trial.suggest_float(
        "focal_gamma",
        1.0,
        2.5
    )

    model = EmotionModel(
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        ff_mult=ff_mult
    ).to(device)

    model = train_model(
        model,
        train_loader,
        val_loader,
        lr=lr,
        weight_decay=weight_decay,
        epochs=OPTUNA_EPOCHS,
        pos_weight_scale=pos_weight_scale,
        label_smoothing=label_smoothing,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        verbose=False
    )

    val_raw = evaluate(
        model,
        val_loader
    )

    thresholds = optimize_thresholds(
        val_raw["labels"],
        val_raw["probs"]
    )

    val_result = evaluate(
        model,
        val_loader,
        thresholds
    )

    score = objective_score(val_result)

    trial.set_user_attr("val_macro_f1", float(val_result["macro_f1"]))
    trial.set_user_attr("val_micro_f1", float(val_result["micro_f1"]))
    trial.set_user_attr("val_min_per_label_f1", float(val_result["min_per_label_f1"]))
    trial.set_user_attr("val_subset_accuracy", float(val_result["subset_accuracy"]))
    trial.set_user_attr("thresholds", thresholds.tolist())

    return score


# %%

sampler = optuna.samplers.TPESampler(
    seed=SEED,
    multivariate=True
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler
)

# ใส่ค่า baseline/non-optimized configuration เป็น trial แรก
# เพื่อให้ optimization มีจุดเริ่มต้นที่ไม่แพ้ baseline ง่าย ๆ
study.enqueue_trial({
    "depth": 1,
    "heads": 2,
    "attn_dropout": 0.20,
    "ff_dropout": 0.20,
    "ff_mult": 2,
    "lr": 2e-5,
    "weight_decay": 1e-2,
    "pos_weight_scale": 0.0,
    "label_smoothing": 0.0,
    "loss_name": "bce",
    "focal_gamma": 2.0
})

study.optimize(
    objective,
    n_trials=N_TRIALS
)


# %%

print("\n======================")
print("BEST PARAMETERS")
print("======================")

print(study.best_params)

print("\n======================")
print("BEST TRIAL VALIDATION ATTRS")
print("======================")

print("objective_score:", study.best_value)
print("val_macro_f1:", study.best_trial.user_attrs.get("val_macro_f1"))
print("val_micro_f1:", study.best_trial.user_attrs.get("val_micro_f1"))
print("val_min_per_label_f1:", study.best_trial.user_attrs.get("val_min_per_label_f1"))
print("val_subset_accuracy:", study.best_trial.user_attrs.get("val_subset_accuracy"))

best_params = study.best_params


# %%

final_model = EmotionModel(
    depth=best_params["depth"],
    heads=best_params["heads"],
    attn_dropout=best_params["attn_dropout"],
    ff_dropout=best_params["ff_dropout"],
    ff_mult=best_params["ff_mult"]
).to(device)


# %%

final_model = train_model(
    final_model,
    train_loader,
    val_loader,
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    epochs=EPOCHS,
    pos_weight_scale=best_params["pos_weight_scale"],
    label_smoothing=best_params["label_smoothing"],
    loss_name=best_params["loss_name"],
    focal_gamma=best_params["focal_gamma"],
    verbose=True
)


# %%

val_result = evaluate(
    final_model,
    val_loader
)

best_thresholds = optimize_thresholds(
    val_result["labels"],
    val_result["probs"]
)

val_result_tuned = evaluate(
    final_model,
    val_loader,
    best_thresholds
)

print("\n======================")
print("BEST THRESHOLDS FROM FINAL VALIDATION")
print("======================")

for label, thr in zip(
    LABEL_COLUMNS,
    best_thresholds
):
    print(f"{label}: {thr:.3f}")

print("\nVALIDATION RESULT AFTER THRESHOLD TUNING")
print("macro_f1:", val_result_tuned["macro_f1"])
print("micro_f1:", val_result_tuned["micro_f1"])
print("min_per_label_f1:", val_result_tuned["min_per_label_f1"])
print("subset_accuracy:", val_result_tuned["subset_accuracy"])


# %%

def summarize_result(name, result):

    return {
        "split": name,
        "subset_accuracy_exact_match": result["subset_accuracy"],
        "micro_precision": result["micro_precision"],
        "micro_recall": result["micro_recall"],
        "micro_f1": result["micro_f1"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "min_per_label_f1": result["min_per_label_f1"],
        "hamming_loss": result["hamming_loss"]
    }


train_result = evaluate(
    final_model,
    train_eval_loader,
    best_thresholds
)

val_result_final = evaluate(
    final_model,
    val_loader,
    best_thresholds
)

test_result = evaluate(
    final_model,
    test_loader,
    best_thresholds
)

metrics_summary = pd.DataFrame([
    summarize_result("train", train_result),
    summarize_result("validation", val_result_final),
    summarize_result("test", test_result)
])

print("\n======================")
print("FINAL METRICS SUMMARY")
print("======================")
display(metrics_summary)

print("\n======================")
print("BEST THRESHOLDS")
print("======================")

threshold_df = pd.DataFrame({
    "label": LABEL_COLUMNS,
    "threshold": best_thresholds
})

display(threshold_df)

print("\n======================")
print("TEST SET MAIN RESULT")
print("======================")

print(f"Subset Accuracy / Exact Match : {test_result['subset_accuracy']:.4f}")
print(f"Micro Precision              : {test_result['micro_precision']:.4f}")
print(f"Micro Recall                 : {test_result['micro_recall']:.4f}")
print(f"Micro F1                     : {test_result['micro_f1']:.4f}")
print(f"Macro Precision              : {test_result['macro_precision']:.4f}")
print(f"Macro Recall                 : {test_result['macro_recall']:.4f}")
print(f"Macro F1                     : {test_result['macro_f1']:.4f}")
print(f"Weighted F1                  : {test_result['weighted_f1']:.4f}")
print(f"Min Per-label F1             : {test_result['min_per_label_f1']:.4f}")
print(f"Hamming Loss                 : {test_result['hamming_loss']:.4f}")


# %%

print("\n======================")
print("TEST CLASSIFICATION REPORT")
print("======================")

report_dict = classification_report(
    test_result["labels"],
    test_result["preds"],
    target_names=LABEL_COLUMNS,
    zero_division=0,
    output_dict=True
)

per_label_report = pd.DataFrame(report_dict).T
display(per_label_report)

per_label_metrics = pd.DataFrame({
    "label": LABEL_COLUMNS,
    "precision": test_result["per_label_precision"],
    "recall": test_result["per_label_recall"],
    "f1_score": test_result["per_label_f1"],
    "true_support": test_result["labels"].sum(axis=0).astype(int),
    "predicted_positive": test_result["preds"].sum(axis=0).astype(int),
    "threshold": best_thresholds
})

print("\n======================")
print("PER-LABEL TEST METRICS")
print("======================")
display(per_label_metrics)

print("\n======================")
print("TEST MULTI-LABEL CONFUSION MATRIX")
print("======================")

mcm = multilabel_confusion_matrix(
    test_result["labels"],
    test_result["preds"]
)

confusion_rows = []

for i, label in enumerate(LABEL_COLUMNS):
    tn, fp, fn, tp = mcm[i].ravel()

    confusion_rows.append({
        "label": label,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })

confusion_df = pd.DataFrame(confusion_rows)
display(confusion_df)

test_probs_df = pd.DataFrame(
    test_result["probs"],
    columns=[f"prob_{label}" for label in LABEL_COLUMNS]
)

test_preds_df = pd.DataFrame(
    test_result["preds"],
    columns=[f"pred_{label}" for label in LABEL_COLUMNS]
)

test_true_df = test_df[[TEXT_COLUMN] + LABEL_COLUMNS].copy()

test_predictions_output = pd.concat(
    [
        test_true_df.reset_index(drop=True),
        test_probs_df.reset_index(drop=True),
        test_preds_df.reset_index(drop=True)
    ],
    axis=1
)

metrics_summary.to_csv(
    "/content/xlmr_sied_original_metrics_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

threshold_df.to_csv(
    "/content/xlmr_sied_original_thresholds.csv",
    index=False,
    encoding="utf-8-sig"
)

per_label_report.to_csv(
    "/content/xlmr_sied_original_per_label_report.csv",
    encoding="utf-8-sig"
)

per_label_metrics.to_csv(
    "/content/xlmr_sied_original_per_label_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

confusion_df.to_csv(
    "/content/xlmr_sied_original_confusion_matrix.csv",
    index=False,
    encoding="utf-8-sig"
)

test_predictions_output.to_csv(
    "/content/xlmr_sied_original_test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nEXPORTED FILES")
print("/content/xlmr_sied_original_metrics_summary.csv")
print("/content/xlmr_sied_original_thresholds.csv")
print("/content/xlmr_sied_original_per_label_report.csv")
print("/content/xlmr_sied_original_per_label_metrics.csv")
print("/content/xlmr_sied_original_confusion_matrix.csv")
print("/content/xlmr_sied_original_test_predictions.csv")


# %%

SAVE_PATH = "/content/final_xlmr_sied_original_xtransformers_strong_optimized.pt"

torch.save(
    {
        "model_state_dict": final_model.state_dict(),
        "thresholds": best_thresholds,
        "best_params": best_params,
        "labels": LABEL_COLUMNS,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "dataset": "SIED-Thai original/unbalanced (2,400 rows)",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optuna_trials": N_TRIALS,
        "optuna_epochs": OPTUNA_EPOCHS,
        "objective": "0.45 macro_f1 + 0.25 micro_f1 + 0.20 min_per_label_f1 + 0.10 subset_accuracy",
        "note": "Encoder comparison version; encoder frozen; same Strong Optimized EAQD/x-transformers framework."
    },
    SAVE_PATH
)

print("\nMODEL SAVED:", SAVE_PATH)
