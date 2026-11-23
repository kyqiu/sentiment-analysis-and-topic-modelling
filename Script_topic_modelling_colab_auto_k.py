# -*- coding: utf-8 -*-
"""German-language LDA topic modelling for Google Colab.

COLAB SETUP — run this in a separate notebook cell (not in this .py file):
    !pip -q install gensim openpyxl nltk matplotlib seaborn wordcloud

Then run this script from a Colab cell, for example:
    %run /content/drive/MyDrive/sentimentproject/topic_modelling_colab_auto_k.py

The script evaluates K=2..8, retains ceil(30% of the candidates ranked by c_v
coherence, then selects the retained model with the lowest Gensim log perplexity.
Before running, check FILE_PATH, SHEET_NAME, and TEXT_COLUMN below.
TEXT_COLUMN=None means use the first Excel column; otherwise provide a column
name (str) or zero-based column index (int).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional, Union
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud


# ------------------------- User configuration -------------------------
FILE_PATH = "/content/drive/MyDrive/sentimentproject/topic2223DE_labeledtexts_results.xlsx"
SHEET_NAME = "neutral"
TEXT_COLUMN: Optional[Union[str, int]] = None  # None = first column

MIN_TOPICS = 2
MAX_TOPICS = 8
EVALUATION_PASSES = 15
FINAL_PASSES = 30
TOP_COHERENCE_FRACTION = 0.30
OUTPUT_DIR = "/content/drive/MyDrive/sentimentproject/topic_visualizations"
OUTPUT_XLSX = "/content/drive/MyDrive/sentimentproject/german_lda_full_results.xlsx"
OUTPUT_ZIP = "/content/drive/MyDrive/sentimentproject/german_lda_final_results_neutral.zip"
RANDOM_SEED = 42
# ----------------------------------------------------------------------


class GermanLDAModelEvaluator:
    """Preprocess German texts, evaluate LDA topic counts, and plot results."""

    def __init__(self) -> None:
        try:
            german_stopwords = stopwords.words("german")
        except LookupError:
            if not nltk.download("stopwords", quiet=True):
                raise RuntimeError(
                    "Could not download NLTK stopwords. Run nltk.download('stopwords') in Colab."
                )
            german_stopwords = stopwords.words("german")

        self.stop_words = set(german_stopwords)
        self.stop_words.update(
            ["dass", "mehr", "viel", "wenig", "ganz", "immer", "ohne", "dann", "noch"]
        )
        self.stemmer = SnowballStemmer("german")
        self.evaluation_results = {
            "num_topics": [],
            "coherence": [],
            "log_perplexity": [],
            "ambiguity": [],
        }

    def preprocess_text(self, text: object) -> list[str]:
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", " ", text)
        words = text.split()
        words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return [word for word in words if len(word) > 2]

    def prepare_corpus(self, texts: Iterable[object]):
        processed = [self.preprocess_text(text) for text in texts]
        processed = [tokens for tokens in processed if tokens]
        if len(processed) < 2:
            raise ValueError(
                "Need at least 2 non-empty documents after preprocessing. "
                "Check the selected Excel text column and corpus contents."
            )

        # Use the original threshold first; fall back for small/sparse corpora.
        dictionary = corpora.Dictionary(processed)
        dictionary.filter_extremes(no_below=5, no_above=0.7)
        if len(dictionary) == 0:
            dictionary = corpora.Dictionary(processed)
            dictionary.filter_extremes(no_below=1, no_above=1.0)
        if len(dictionary) < 2:
            raise ValueError(
                "Fewer than 2 usable terms remain after preprocessing. "
                "Check that the selected column contains German text."
            )

        corpus = [dictionary.doc2bow(tokens) for tokens in processed]
        # Remove documents made empty by dictionary filtering.
        kept = [(tokens, bow) for tokens, bow in zip(processed, corpus) if bow]
        processed = [tokens for tokens, _ in kept]
        corpus = [bow for _, bow in kept]
        if len(corpus) < 2:
            raise ValueError("Fewer than 2 documents contain usable terms after filtering.")

        self.dictionary = dictionary
        self.corpus = corpus
        self.processed_texts = processed
        return corpus, dictionary

    def calculate_ambiguity(self, lda_model) -> float:
        """Mean document-topic entropy; higher values mean more diffuse mixtures."""
        entropies = []
        for doc in self.corpus:
            probabilities = [p for _, p in lda_model.get_document_topics(doc, minimum_probability=0)]
            entropies.append(-sum(p * math.log2(p) for p in probabilities if p > 0))
        return sum(entropies) / len(entropies)

    def evaluate_topic_range(
        self,
        min_topics: int = MIN_TOPICS,
        max_topics: int = MAX_TOPICS,
        passes: int = EVALUATION_PASSES,
    ) -> dict:
        if not hasattr(self, "corpus"):
            raise ValueError("Call prepare_corpus() before evaluating topic counts.")
        if min_topics < 2 or max_topics < min_topics:
            raise ValueError("Topic range must satisfy 2 <= min_topics <= max_topics.")

        max_topics = min(max_topics, len(self.dictionary))
        if max_topics < min_topics:
            raise ValueError("The vocabulary is too small for the requested topic range.")

        # Clear previous results if evaluation is run a second time.
        for values in self.evaluation_results.values():
            values.clear()

        print(f"Evaluating topic counts from {min_topics} to {max_topics}...\n")
        for num_topics in range(min_topics, max_topics + 1):
            lda_model = models.LdaModel(
                self.corpus,
                num_topics=num_topics,
                id2word=self.dictionary,
                passes=passes,
                random_state=RANDOM_SEED,
            )
            coherence = models.CoherenceModel(
                model=lda_model,
                texts=self.processed_texts,
                dictionary=self.dictionary,
                coherence="c_v",
            ).get_coherence()
            log_perplexity = lda_model.log_perplexity(self.corpus)
            ambiguity = self.calculate_ambiguity(lda_model)

            self.evaluation_results["num_topics"].append(num_topics)
            self.evaluation_results["coherence"].append(coherence)
            self.evaluation_results["log_perplexity"].append(log_perplexity)
            self.evaluation_results["ambiguity"].append(ambiguity)
            print(
                f"Topics: {num_topics:2d} | Coherence: {coherence:.4f} | "
                f"Log perplexity: {log_perplexity:.4f} | Ambiguity: {ambiguity:.4f}"
            )
        return self.evaluation_results

    def select_optimal_topics(self, top_fraction: float = TOP_COHERENCE_FRACTION) -> int:
        """Keep ceil(top_fraction) by coherence, then minimize log perplexity."""
        if not self.evaluation_results["num_topics"]:
            raise ValueError("Run evaluate_topic_range() before selecting K.")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be greater than 0 and at most 1.")

        candidates = []
        for i, num_topics in enumerate(self.evaluation_results["num_topics"]):
            coherence = self.evaluation_results["coherence"][i]
            log_perplexity = self.evaluation_results["log_perplexity"][i]
            if math.isfinite(coherence) and math.isfinite(log_perplexity):
                candidates.append({
                    "num_topics": num_topics,
                    "coherence": coherence,
                    "log_perplexity": log_perplexity,
                })
        if not candidates:
            raise ValueError("No valid candidate models are available for K selection.")

        keep_n = max(1, math.ceil(len(candidates) * top_fraction))
        ranked = sorted(candidates, key=lambda row: (-row["coherence"], row["num_topics"]))
        retained = ranked[:keep_n]
        selected = min(
            retained,
            key=lambda row: (row["log_perplexity"], -row["coherence"], row["num_topics"]),
        )
        retained_keys = {row["num_topics"] for row in retained}
        for rank, row in enumerate(ranked, start=1):
            row["coherence_rank"] = rank
            row["retained_top_fraction"] = row["num_topics"] in retained_keys
            row["selected"] = row["num_topics"] == selected["num_topics"]
        self.k_selection_results = ranked
        self.optimal_topics = selected["num_topics"]
        self.top_fraction = top_fraction

        print(
            f"K selection: retained {keep_n}/{len(candidates)} candidates "
            f"(ceil({len(candidates)} x {top_fraction:.0%})); "
            f"selected K={self.optimal_topics} with the lowest log perplexity "
            "among the retained models."
        )
        return self.optimal_topics

    def plot_evaluation_results(self, output_dir: Path) -> None:
        if not self.evaluation_results["num_topics"]:
            raise ValueError("Run evaluate_topic_range() before plotting results.")

        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        series = [
            ("coherence", "Coherence (higher is generally better)"),
            ("log_perplexity", "Gensim log perplexity (compare across models)"),
            ("ambiguity", "Mean topic entropy (higher = more diffuse)"),
        ]
        for ax, (key, label) in zip(axes, series):
            ax.plot(
                self.evaluation_results["num_topics"],
                self.evaluation_results[key],
                marker="o",
            )
            ax.set_xlabel("Number of topics")
            ax.set_ylabel(label)
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(output_dir / "topic_evaluation_metrics.png", dpi=300)
        plt.show()
        plt.close(fig)

    def train_best_model(self, num_topics: int, passes: int = FINAL_PASSES):
        if not hasattr(self, "corpus"):
            raise ValueError("Call prepare_corpus() before training the final model.")
        if num_topics < 2 or num_topics > len(self.dictionary):
            raise ValueError(
                f"num_topics must be between 2 and {len(self.dictionary)} for this vocabulary."
            )
        self.best_lda_model = models.LdaModel(
            self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=passes,
            random_state=RANDOM_SEED,
            per_word_topics=True,
        )
        return self.best_lda_model

    def print_topics(self, num_words: int = 10) -> None:
        if not hasattr(self, "best_lda_model"):
            raise ValueError("Train the final model before printing topics.")
        for topic_id, topic in self.best_lda_model.print_topics(num_words=num_words):
            print(f"Topic #{topic_id}: {topic}")

    def plot_topic_keywords(self, output_dir: Path, num_words: int = 10) -> None:
        if not hasattr(self, "best_lda_model"):
            raise ValueError("Train the final model before plotting topics.")

        num_topics = self.best_lda_model.num_topics
        rows = (num_topics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows), squeeze=False)
        axes = axes.flatten()
        for topic_id in range(num_topics):
            terms = self.best_lda_model.get_topic_terms(topic_id, topn=num_words)
            words = [self.dictionary[term_id] for term_id, _ in terms]
            weights = [weight for _, weight in terms]
            axes[topic_id].barh(words, weights)
            axes[topic_id].set_title(f"Topic #{topic_id}: Top {num_words} terms")
            axes[topic_id].set_xlabel("Weight")
        for ax in axes[num_topics:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "topic_keywords_bar.png", dpi=300)
        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(rows, 2, figsize=(15, 7 * rows), squeeze=False)
        axes = axes.flatten()
        for topic_id in range(num_topics):
            terms = self.best_lda_model.get_topic_terms(topic_id, topn=num_words)
            frequencies = {self.dictionary[term_id]: weight for term_id, weight in terms}
            cloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=num_words,
                relative_scaling=0.8,
            ).generate_from_frequencies(frequencies)
            axes[topic_id].imshow(cloud, interpolation="bilinear")
            axes[topic_id].set_title(f"Topic #{topic_id}: Word cloud")
            axes[topic_id].axis("off")
        for ax in axes[num_topics:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "topic_keywords_wordcloud.png", dpi=300)
        plt.show()
        plt.close(fig)


def load_texts() -> tuple[list[str], int, int]:
    file_path = Path(FILE_PATH)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Excel file not found: {file_path}\n"
            "Confirm FILE_PATH and mount Google Drive at /content/drive."
        )

    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    if df.empty:
        raise ValueError(f"Worksheet '{SHEET_NAME}' contains no rows.")

    if TEXT_COLUMN is None:
        selected = df.iloc[:, 0]
        selected_name = str(df.columns[0])
    elif isinstance(TEXT_COLUMN, int):
        if TEXT_COLUMN < 0 or TEXT_COLUMN >= len(df.columns):
            raise IndexError(f"TEXT_COLUMN={TEXT_COLUMN} is outside the worksheet columns.")
        selected = df.iloc[:, TEXT_COLUMN]
        selected_name = str(df.columns[TEXT_COLUMN])
    else:
        if TEXT_COLUMN not in df.columns:
            raise KeyError(
                f"Column '{TEXT_COLUMN}' not found. Available columns: {list(df.columns)}"
            )
        selected = df[TEXT_COLUMN]
        selected_name = TEXT_COLUMN

    total_rows = len(df)
    texts = [str(value).strip() for value in selected if pd.notna(value) and str(value).strip()]
    print(
        f"Read {len(df)} rows from sheet '{SHEET_NAME}'; "
        f"using column '{selected_name}', {len(texts)} non-empty texts."
    )
    if len(texts) < 2:
        raise ValueError("Fewer than 2 non-empty texts found in the selected column.")
    return texts, total_rows, len(texts)


def export_results(evaluator: GermanLDAModelEvaluator, total_rows: int, valid_texts: int, output_dir: Path) -> Path:
    """Write the three requested result sheets and a ZIP containing the workbook and PNGs."""
    if not hasattr(evaluator, "best_lda_model"):
        raise ValueError("Train the final model before exporting results.")

    evaluation_df = pd.DataFrame({
        "num_topics": evaluator.evaluation_results["num_topics"],
        "coherence": evaluator.evaluation_results["coherence"],
        # Gensim's log_perplexity is retained under the sample workbook's column name.
        "perplexity": evaluator.evaluation_results["log_perplexity"],
        "ambiguity": evaluator.evaluation_results["ambiguity"],
    })

    keyword_rows = []
    for topic_id in range(evaluator.best_lda_model.num_topics):
        for rank, (word, weight) in enumerate(
            evaluator.best_lda_model.show_topic(topic_id, topn=15), start=1
        ):
            keyword_rows.append({
                "topic_id": topic_id,
                "keyword_rank": rank,
                "keyword": word,
                "weight": weight,
            })
    keywords_df = pd.DataFrame(
        keyword_rows, columns=["topic_id", "keyword_rank", "keyword", "weight"]
    )

    stats_df = pd.DataFrame([
        ("Total Input Texts", total_rows),
        ("Valid Texts After Filtering", valid_texts),
        ("Vocabulary Size (After Filtering)", len(evaluator.dictionary)),
        ("Optimal Number of Topics", evaluator.best_lda_model.num_topics),
        ("K Selection Rule", f"Top {evaluator.top_fraction:.0%} by coherence (ceiling), then lowest log perplexity"),
        ("Number of Candidate Models Retained", sum(row["retained_top_fraction"] for row in evaluator.k_selection_results)),
        ("Number of Documents in Corpus", len(evaluator.corpus)),
        ("Random Seed Used", RANDOM_SEED),
    ], columns=["Metric", "Value"])

    xlsx_path = Path(OUTPUT_XLSX)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        evaluation_df.to_excel(writer, sheet_name="Topic_Evaluation_Results", index=False)
        keywords_df.to_excel(writer, sheet_name="Best_Model_Keywords", index=False)
        stats_df.to_excel(writer, sheet_name="Preprocessing_Stats", index=False)
        pd.DataFrame(evaluator.k_selection_results).to_excel(
            writer, sheet_name="K_Selection", index=False
        )

    zip_path = Path(OUTPUT_ZIP)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    chart_names = [
        "topic_evaluation_metrics.png",
        "topic_keywords_bar.png",
        "topic_keywords_wordcloud.png",
    ]
    missing = [name for name in chart_names if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Expected visualization file(s) missing: {missing}")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for name in chart_names:
            archive.write(output_dir / name, arcname=f"topic_visualizations/{name}")
        archive.write(xlsx_path, arcname=xlsx_path.name)

    print(f"Excel workbook saved to: {xlsx_path}")
    print(f"ZIP archive saved to: {zip_path}")
    return zip_path


def mount_google_drive() -> None:
    """Mount Drive in Colab; raise a useful message if run outside Colab."""
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "This script is configured for Google Colab. Run it in a Colab runtime, "
            "or change FILE_PATH to a local Excel path and adapt Drive mounting."
        ) from exc
    drive.mount("/content/drive")


def main() -> None:
    mount_google_drive()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, total_rows, valid_texts = load_texts()
    evaluator = GermanLDAModelEvaluator()
    corpus, dictionary = evaluator.prepare_corpus(texts)
    print(f"Preprocessing complete: {len(corpus)} documents, {len(dictionary)} terms.\n")

    evaluator.evaluate_topic_range(
        min_topics=MIN_TOPICS,
        max_topics=MAX_TOPICS,
        passes=EVALUATION_PASSES,
    )
    evaluator.plot_evaluation_results(output_dir)

    optimal_topics = evaluator.select_optimal_topics(TOP_COHERENCE_FRACTION)
    evaluator.train_best_model(optimal_topics, passes=FINAL_PASSES)
    print("\nFinal model topic keywords:")
    evaluator.print_topics(num_words=8)
    evaluator.plot_topic_keywords(output_dir, num_words=15)
    export_results(evaluator, total_rows, valid_texts, output_dir)
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    main()
