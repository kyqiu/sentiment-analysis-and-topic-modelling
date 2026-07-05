# ==================== 【Colab第1格】安装依赖 ====================
!pip install gensim pandas openpyxl nltk -q

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import warnings
import os
from google.colab import files

warnings.filterwarnings("ignore")
nltk.download("punkt")

# 创建模型保存文件夹
os.makedirs("lda_trained_models", exist_ok=True)
print("✅ 环境安装完成")

#===============#
# ==================== 【Colab第2格】上传Excel文件 ====================
print("请选择你的Excel文件（.xlsx格式）")
uploaded = files.upload()

# 获取上传的文件名
file_name = list(uploaded.keys())[0]
print(f"\n✅ 已上传文件: {file_name}")

# 读取Excel（默认读取第一个sheet）
df = pd.read_excel(file_name)

# ========== 重要：修改这里匹配你的Excel列名 ==========
# 把下面两个引号里的内容，改成你Excel里实际的列名
TEXT_COLUMN = "text"           # 文本列的列名
SENTIMENT_COLUMN = "sentiment" # 情感标签列的列名
# =====================================================

# 重命名为统一列名，方便后续代码运行
df = df.rename(columns={TEXT_COLUMN: "text", SENTIMENT_COLUMN: "sentiment_label"})

print(f"\n数据总条数: {len(df)}")
print("\n情感标签分布：")
print(df["sentiment_label"].value_counts())
print("\n前3条数据预览：")
print(df[["text", "sentiment_label"]].head(3))

#=========================#

# ==================== 【Colab第3格】LDA建模主程序 ====================
# 固定5类情感标签（和你的标注体系一致）
target_sentiments = ["worried", "critical", "hopeful", "happy", "neutral"]

# 过滤只保留目标5类文本
df = df[df["sentiment_label"].isin(target_sentiments)].reset_index(drop=True)
print(f"过滤后有效数据条数: {len(df)}")

# ---------- 文本预处理函数 ----------
# 【德语/英文】用这个
def preprocess_text(raw_text):
    tokens = word_tokenize(str(raw_text).lower())
    tokens = [t for t in tokens if len(t) > 2 and t.isalpha()]
    return tokens

# 【中文】把上面函数注释掉，用下面这个（先安装jieba）
# !pip install jieba -q
# import jieba
# def preprocess_text(raw_text):
#     tokens = jieba.lcut(str(raw_text))
#     tokens = [t for t in tokens if len(t) > 1]
#     return tokens

# 批量预处理
df["tokens"] = df["text"].apply(preprocess_text)

# 候选主题K值范围（根据数据量调整，数据多可以扩大到2~30）
k_candidates = list(range(2, 16))

# 存储结果
final_results = {}
all_metrics_records = []

# ---------- 按5类情感分别训练LDA ----------
for sentiment in target_sentiments:
    print(f"\n{'='*60}")
    print(f"  正在处理情感子语料: {sentiment}")
    print(f"{'='*60}")
    
    # 筛选当前情感子语料
    sub_df = df[df["sentiment_label"] == sentiment].copy()
    token_corpus = sub_df["tokens"].tolist()
    print(f"子语料文本数量: {len(token_corpus)}")
    
    if len(token_corpus) < 5:
        print(f"⚠️  {sentiment} 数据量太少（<5条），跳过建模")
        continue
    
    # 构建词典和词袋
    dictionary = corpora.Dictionary(token_corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in token_corpus]
    
    model_metrics = []
    
    # 遍历所有K值训练
    for k in k_candidates:
        lda_model = models.LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=15,
            alpha="auto",
            per_word_topics=True
        )
        
        # 计算Topic Coherence (c_v)
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=token_corpus,
            dictionary=dictionary,
            coherence="c_v"
        )
        coherence_score = coherence_model.get_coherence()
        
        # 计算Perplexity
        perplexity_score = lda_model.log_perplexity(bow_corpus)
        
        record = {
            "sentiment": sentiment,
            "K": k,
            "coherence": coherence_score,
            "perplexity": perplexity_score
        }
        model_metrics.append(record)
        all_metrics_records.append(record)
    
    # 打印该情感所有K的指标
    print("\n全部候选K指标：")
    df_m = pd.DataFrame(model_metrics)
    print(df_m.sort_values("coherence", ascending=False).to_string(index=False))
    
    # ---------- 论文筛选规则：Coherence前30% → 最低Perplexity ----------
    df_metrics = pd.DataFrame(model_metrics)
    df_metrics = df_metrics.sort_values("coherence", ascending=False).reset_index(drop=True)
    total_candidates = len(df_metrics)
    top_30_pct_num = max(1, int(total_candidates * 0.3))
    top_coherence_models = df_metrics.iloc[:top_30_pct_num]
    
    best_row = top_coherence_models.loc[top_coherence_models["perplexity"].idxmin()]
    best_k = int(best_row["K"])
    best_coh = best_row["coherence"]
    best_ppl = best_row["perplexity"]
    
    # 重新训练最优模型并保存
    best_lda = models.LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=best_k,
        random_state=42,
        passes=15,
        alpha="auto",
        per_word_topics=True
    )
    
    save_path = f"lda_trained_models/lda_{sentiment}_k{best_k}.model"
    best_lda.save(save_path)
    
    print(f"\n🏆 最优结果: K={best_k}")
    print(f"   Coherence = {best_coh:.4f}")
    print(f"   Perplexity = {best_ppl:.4f}")
    print(f"\n各主题Top关键词：")
    for topic_id, topic_words in best_lda.print_topics(num_words=8):
        print(f"  Topic {topic_id}: {topic_words}")
    
    # 人工核验提示
    print(f"\n📝 人工核验备选（Coherence前30%）：")
    print(top_coherence_models[["K", "coherence", "perplexity"]].to_string(index=False))
    
    final_results[sentiment] = {
        "optimal_K": best_k,
        "lda_model": best_lda,
        "coherence": best_coh,
        "perplexity": best_ppl,
        "dictionary": dictionary,
        "bow_corpus": bow_corpus,
        "model_path": save_path
    }

    #========================#

    # ==================== 【Colab第4格】导出结果并下载 ====================
# 1. 导出所有K值指标CSV
metrics_df = pd.DataFrame(all_metrics_records)
metrics_df.to_csv("lda_all_k_metrics.csv", index=False, encoding="utf-8-sig")
print("✅ 全部K值指标已保存: lda_all_k_metrics.csv")

# 2. 导出最终最优配置汇总
summary_data = []
for sent, res in final_results.items():
    summary_data.append({
        "情感标签": sent,
        "最优主题数K": res["optimal_K"],
        "Topic Coherence": round(res["coherence"], 4),
        "Log Perplexity": round(res["perplexity"], 4)
    })
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("lda_optimal_summary.csv", index=False, encoding="utf-8-sig")
print("✅ 最优配置汇总已保存: lda_optimal_summary.csv")
print("\n📊 最终结果汇总：")
print(summary_df.to_string(index=False))

# 3. 打包所有模型文件下载
!zip -r lda_trained_models.zip lda_trained_models/
print("\n✅ 模型文件已打包: lda_trained_models.zip")

# 4. 自动下载到本地
files.download("lda_all_k_metrics.csv")
files.download("lda_optimal_summary.csv")
files.download("lda_trained_models.zip")








# ==================== 【Colab第4格】导出结果并下载 ====================
# 1. 导出所有K值指标CSV
metrics_df = pd.DataFrame(all_metrics_records)
metrics_df.to_csv("lda_all_k_metrics.csv", index=False, encoding="utf-8-sig")
print("✅ 全部K值指标已保存: lda_all_k_metrics.csv")

# 2. 导出最终最优配置汇总
summary_data = []
for sent, res in final_results.items():
    summary_data.append({
        "情感标签": sent,
        "最优主题数K": res["optimal_K"],
        "Topic Coherence": round(res["coherence"], 4),
        "Log Perplexity": round(res["perplexity"], 4)
    })
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("lda_optimal_summary.csv", index=False, encoding="utf-8-sig")
print("✅ 最优配置汇总已保存: lda_optimal_summary.csv")
print("\n📊 最终结果汇总：")
print(summary_df.to_string(index=False))

# 3. 打包所有模型文件下载
!zip -r lda_trained_models.zip lda_trained_models/
print("\n✅ 模型文件已打包: lda_trained_models.zip")

# 4. 自动下载到本地
files.download("lda_all_k_metrics.csv")
files.download("lda_optimal_summary.csv")
files.download("lda_trained_models.zip")