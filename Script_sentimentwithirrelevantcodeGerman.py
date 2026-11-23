# ============================================================
# 多语种 BERT 六分类情感分类
# 输出每类及 micro/macro/weighted 平均指标
# ============================================================

import os

# 必须在导入 TensorFlow 和任何 transformers 模块前设置
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tensorflow as tf
    import tf_keras as keras
    print("TensorFlow version:", tf.__version__)
except ImportError as e:
    print("TensorFlow 或 tf_keras 导入失败，请运行 pip 安装单元格并重启运行时。")
    raise e

import json
import numpy as np
import pandas as pd
import transformers

from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# 确保 tensorflow 导入后再导入 transformers 模型
from transformers import BertTokenizer, TFBertForSequenceClassification


# =========================
# 1. 配置
# =========================

TEXT_COL = "text"
LABEL_COL = "label"

LABELS = [
    "critical",
    "hopeful",
    "happiness",
    "worried",
    "neutral",
    "irrelevant",
]

MODEL_NAME = "bert-base-German-cased"

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

EXCEL_PATH = (
    "/content/drive/MyDrive/sentimentproject/"
    "2026dataSWISScombined_data2024most.xlsx"
)

OUTPUT_PARENT_DIR = os.path.dirname(EXCEL_PATH)
MODEL_OUTPUT_DIR = os.path.join(
    OUTPUT_PARENT_DIR,
    "saved_bert_model_DE",
)


# =========================
# 2. 初始化、检查 GPU、挂载 Drive
# =========================

print("Transformers version:", transformers.__version__)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

drive.mount("/content/drive")

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow detected GPUs:", gpus)
print("训练设备：", "GPU" if gpus else "CPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU memory growth 设置提示：", e)


# =========================
# 3. 读取和检查 Excel
# =========================

if not os.path.isfile(EXCEL_PATH):
    raise FileNotFoundError(
        f"找不到 Excel 文件：{EXCEL_PATH}\n"
        "请检查文件路径 and 文件名。"
    )

print("\n读取文件：", EXCEL_PATH)
df = pd.read_excel(EXCEL_PATH)

print("Excel 列名：", list(df.columns))

for col in [TEXT_COL, LABEL_COL]:
    if col not in df.columns:
        raise KeyError(
            f"Excel 中找不到列 {col!r}。"
            f"当前列名：{list(df.columns)}"
        )

df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

# 自动修正 Excel 中常见的拼写错误
df[LABEL_COL] = df[LABEL_COL].replace({
    "irrelevent": "irrelevant"
})

# 删除空文本，并保留 Excel 中原始行号，方便追溯
df = df[df[TEXT_COL] != ""].copy()
df = df.reset_index(drop=False).rename(
    columns={"index": "source_row"}
)

unknown_labels = sorted(set(df[LABEL_COL]) - set(LABELS))
if unknown_labels:
    raise ValueError(
        f"数据中存在预设标签之外的标签：{unknown_labels}\n"
        f"预期标签为：{LABELS}"
    )

missing_labels = [
    label for label in LABELS
    if label not in set(df[LABEL_COL])
]
if missing_labels:
    raise ValueError(
        f"数据中缺少以下类别：{missing_labels}。"
        "请检查 Excel 标签列或 LABELS 设置。"
    )

print(f"\n有效段落数：{len(df):,}")
print("各类别样本数：")
print(df[LABEL_COL].value_counts().reindex(LABELS, fill_value=0))

class_counts = df[LABEL_COL].value_counts()
if class_counts.min() < 10:
    raise ValueError(
        "至少有一个类别少于 10 个样本，可能无法可靠地分层切分。"
        f"\n各类别数量：\n{class_counts}"
    )


# =========================
# 4. 分层划分训练、验证、测试集
# =========================

texts = df[TEXT_COL].to_numpy()

label_to_id = {label: i for i, label in enumerate(LABELS)}
id_to_label = {i: label for label, i in label_to_id.items()}

y = df[LABEL_COL].map(label_to_id).to_numpy(dtype=np.int32)
indices = np.arange(len(df))

# 20% 测试集
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y,
)

# 剩余数据中取 12.5% 作为验证集
# 总比例约为：训练 70%、验证 10%、测试 20%
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.125,
    random_state=RANDOM_SEED,
    stratify=y[train_val_idx],
)

print(
    f"\n训练集：{len(train_idx):,}；"
    f"验证集：{len(val_idx):,}；"
    f"测试集：{len(test_idx):,}"
)


# =========================
# 5. 加载 Tokenizer 和模型
# =========================

print("\n加载 Tokenizer 和多语种 BERT 模型……")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

model = TFBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id_to_label,
    label2id=label_to_id,
    from_pt=True,
    use_safetensors=False,
)

print("模型和 Tokenizer 加载成功。")


# =========================
# 6. Tokenize 并建立 TensorFlow 数据集
# =========================

def make_dataset(text_array, label_array, shuffle=False):
    encoded = tokenizer(
        list(text_array),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np",
    )

    features = {
        key: np.asarray(value, dtype=np.int32)
        for key, value in encoded.items()
    }

    labels_array = np.asarray(label_array, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels_array)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(labels_array),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(
    texts[train_idx],
    y[train_idx],
    shuffle=True,
)

val_ds = make_dataset(
    texts[val_idx],
    y[val_idx],
    shuffle=False,
)

test_ds = make_dataset(
    texts[test_idx],
    y[test_idx],
    shuffle=False,
)


# =========================
# 7. 编译并训练
# =========================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    ],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )
]

print("\n开始训练……")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

print("\n训练完成。")


# =========================
# 8. 测试集预测
# =========================

print("\n评估测试集……")

test_evaluation = model.evaluate(
    test_ds,
    return_dict=True,
    verbose=1,
)
print("测试集 loss 和 accuracy：", test_evaluation)

# 获取测试集 logits
logits_list = []

for batch_features, batch_labels in test_ds:
    outputs = model(batch_features, training=False)
    logits_list.append(outputs.logits.numpy())

logits = np.concatenate(logits_list, axis=0)
y_true = y[test_idx]
y_pred = np.argmax(logits, axis=1)

probabilities = tf.nn.softmax(logits, axis=1).numpy()


# =========================
# 9. 计算每类、总体及平均指标
# =========================

label_ids = list(range(len(LABELS)))

# 每个类别分别计算指标
class_precision, class_recall, class_f1, class_support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
)

metric_rows = []

for i, label in enumerate(LABELS):
    metric_rows.append({
        "category": label,
        "precision": class_precision[i],
        "recall": class_recall[i],
        "f1": class_f1[i],
        "accuracy": np.nan,
        "support": int(class_support[i]),
    })

# 总体指标以及 macro / weighted 平均
for average_name in ["micro", "macro", "weighted"]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=average_name,
        zero_division=0,
    )

    metric_rows.append({
        "category": f"{average_name} avg",
        "precision": p,
        "recall": r,
        "f1": f,
        "accuracy": np.nan,
        "support": len(y_true),
    })

# Accuracy 单独列出，不与 precision/recall/F1 混为一谈
overall_accuracy = accuracy_score(y_true, y_pred)

metric_rows.append({
    "category": "accuracy",
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "accuracy": overall_accuracy,
    "support": len(y_true),
})

metrics_df = pd.DataFrame(metric_rows)

print("\n测试集指标：")
print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================
# 10. 保存模型、指标和预测结果
# =========================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 保存微调后的模型和 Tokenizer
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# 保存各类别指标、micro/macro/weighted average 和 accuracy
metrics_path = os.path.join(
    OUTPUT_PARENT_DIR,
    "sentiment_metrics.csv",
)
metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig",
)

# 保存测试集逐轮训练和验证指标
history_df = pd.DataFrame(history.history)
history_df.insert(
    0,
    "epoch",
    np.arange(1, len(history_df) + 1),
)

history_path = os.path.join(
    OUTPUT_PARENT_DIR,
    "training_history.csv",
)
history_df.to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig",
)

# 保存测试集每条文本的真实标签、预测标签和类别概率
test_results = df.iloc[test_idx][
    ["source_row", TEXT_COL, LABEL_COL]
].copy()

test_results = test_results.rename(
    columns={LABEL_COL: "true_label"}
)

test_results["predicted_label"] = [
    id_to_label[int(i)] for i in y_pred
]
test_results["prediction_confidence"] = probabilities.max(axis=1)

for class_id, label in id_to_label.items():
    test_results[f"prob_{label}"] = probabilities[:, class_id]

test_predictions_path = os.path.join(
    OUTPUT_PARENT_DIR,
    "test_predictions.csv",
)
test_results.to_csv(
    test_predictions_path,
    index=False,
    encoding="utf-8-sig",
)

# 保存标签映射和训练参数
with open(
    os.path.join(MODEL_OUTPUT_DIR, "label_mapping.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "label_to_id": label_to_id,
            "id_to_label": {
                str(k): v for k, v in id_to_label.items()
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(
    os.path.join(MODEL_OUTPUT_DIR, "training_config.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "epochs_completed": len(history.history["loss"]),
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "labels": LABELS,
            "split": {
                "train": 0.70,
                "validation": 0.10,
                "test": 0.20,
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n全部完成，文件保存位置：")
print("微调模型和 Tokenizer：", MODEL_OUTPUT_DIR)
print("测试集指标：", metrics_path)
print("训练历史：", history_path)
print("测试集逐条预测：", test_predictions_path)