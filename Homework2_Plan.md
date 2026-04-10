# Homework 2 實作計畫：Detect AI Generated Text

## 1. 作業目標與成功標準

本作業的目標是根據 `Homework 2.pdf` 的要求，完成一套可重現的 AI 生成文本偵測流程，並用同一份資料與驗證集比較傳統機器學習方法與 BERT 模型的表現，最後再用本地 LLM 進行對抗改寫測試。

最終需完成以下四部分：

1. 資料探索與傳統 baseline
2. `bert-base-cased` 與 `bert-large-cased` 微調與比較
3. Local LLM 對抗改寫與 detector 驗證
4. GitHub 程式整理與 PDF 報告

成功標準如下：

- `Baseline.py` 可獨立完成資料切分、TF-IDF 訓練、ROC-AUC 評估與基礎圖表輸出
- `BERT.py` 可用同一份 split 跑 `bert-base-cased` 與 `bert-large-cased`，並輸出 ROC-AUC 與訓練紀錄
- `LocalLLM.py` 可從 validation set 選取 `5-10` 篇 human essay 進行改寫，並用最佳 detector 重新評估
- 報告中至少有一張模型比較表、一組訓練或評估圖、一至兩個 adversarial case 分析
- 全部流程能在 GitHub repo 中被清楚重現

---

## 2. 目前環境與依賴狀態

### 2.1 已確認環境

根據目前工作環境檢查結果：

| 項目 | 狀態 |
|------|------|
| Python | `3.14.3` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| transformers | `5.5.3` |
| datasets | `4.8.4` |
| pandas | `3.0.1` |
| scikit-learn | `1.8.0` |
| CUDA 可用性 | `True` |
| GPU | `NVIDIA GeForce RTX 4090 (24GB)` |

已確認 RTX 4090 可被 PyTorch 正常偵測，且能完成 GPU tensor 運算與 `bert-base-cased` 的最小前向推論。

### 2.2 目前缺少的套件

以下套件目前尚未安裝：

- `accelerate`
- `bitsandbytes`
- `torchaudio`（本作業通常不是必要）

建議安裝方向：

```bash
pip install accelerate
pip install bitsandbytes
```

若 Windows 上 `bitsandbytes` 安裝不穩定，可先以 `fp16` 完成 BERT 與 LLM 實驗，再將量化視為進階優化。

### 2.3 其他非阻塞事項

- 尚未設定 `HF_TOKEN`，下載 Hugging Face 模型時可能遇到較低 rate limit
- Windows 未開啟 symlink / Developer Mode，HF cache 仍可使用，但會更佔空間

---

## 3. 系統架構

整體流程分為五層：

1. `Data Layer`
   負責讀取 `train_v2_drcat_02.csv`、清理欄位、建立固定 split。
2. `Training Layer`
   負責訓練 TF-IDF baseline 與 BERT 模型。
3. `Evaluation Layer`
   負責 ROC-AUC、accuracy、混淆矩陣、訓練紀錄與比較表輸出。
4. `Attack Layer`
   負責使用 Local LLM 改寫 validation set 中的 human essay，再送回 detector。
5. `Reporting Layer`
   負責整合圖表、表格與案例，輸出報告素材。

文字架構圖如下：

```text
train_v2_drcat_02.csv
        |
        v
Data Loading / Fixed Split / EDA
        |
        +--------------------+
        |                    |
        v                    v
TF-IDF + Logistic      BERT Base / Large
Regression Baseline    Fine-tuning
        |                    |
        +---------+----------+
                  |
                  v
        Validation Metrics / Model Comparison
                  |
                  v
       Best Detector + Local LLM Rewrite Attack
                  |
                  v
         Attack Analysis / Report Assets / PDF
```

---

## 4. 專案模組設計

本作業將以三支主腳本加上一個共用工具模組的方式實作，避免腳本之間依賴執行時共享變數。

### 4.1 主腳本責任

| 檔案 | 責任 |
|------|------|
| `Baseline.py` | 資料讀取、固定 split、TF-IDF + Logistic Regression、baseline ROC-AUC |
| `BERT.py` | BERT tokenization、Trainer 訓練、模型保存、validation metrics |
| `LocalLLM.py` | 載入最佳 detector、抽樣 human essay、生成 attack samples、重測 detector |
| `data_utils.py` | 共用資料讀取、split、seed、metrics、artifact 路徑管理 |

### 4.2 為什麼需要共用工具模組

目前 `BERT.py` 與 `LocalLLM.py` 都直接使用未定義的 `X_train`、`X_val`、`y_train`、`y_val`，這代表它們假設自己與 `Baseline.py` 共用命名空間，但實際上三個檔案是獨立執行的。正式版本必須改成以下其中一種結構：

- 每支腳本自行讀取 CSV 並建立相同 split
- 或由共用模組統一提供 `train_df` / `val_df`

建議採第二種，因為這樣可以確保 baseline、BERT、attack 都使用完全一致的 validation set。

### 4.3 建議輸出目錄

```text
artifacts/
  baseline/
  bert_base/
  bert_large/
  attacks/
  report_assets/
```

對應原則如下：

- baseline 模型、vectorizer、EDA 圖與 baseline 預測放在 `artifacts/baseline/`
- `bert-base-cased` 訓練結果放在 `artifacts/bert_base/`
- `bert-large-cased` 訓練結果放在 `artifacts/bert_large/`
- 對抗改寫文本與重測結果放在 `artifacts/attacks/`
- 報告中會直接使用的圖表與比較表放在 `artifacts/report_assets/`

---

## 5. 資料集分析與前處理策略

### 5.1 資料集真實統計

`train_v2_drcat_02.csv` 的實際統計如下：

| 指標 | 數值 |
|------|------|
| 總筆數 | `44,868` |
| 欄位 | `text`, `label`, `prompt_name`, `source`, `RDizzl3_seven` |
| Human (`label=0`) | `27,371` |
| AI (`label=1`) | `17,497` |
| Prompt 數量 | `15` |
| Source 數量 | `17` |

此資料集有輕微類別不平衡，比例約為 `61% : 39%`，但尚未到必須重新抽樣的程度。主要評估指標仍以 ROC-AUC 為主。

### 5.2 模型實際使用的欄位

訓練特徵只使用 `text` 欄位。

以下欄位不直接作為模型輸入：

- `prompt_name`
- `source`
- `RDizzl3_seven`

這些欄位只用於：

- EDA
- split 分層
- 誤差分析
- limitation 討論

### 5.3 Split 策略

建議固定使用 `random_state=42`，並優先採用 `label + prompt_name` 的聯合分層。這樣可以同時穩定類別比例與 prompt 分布。

示意做法：

```python
df = pd.read_csv("train_v2_drcat_02.csv")
split_key = df["label"].astype(str) + "_" + df["prompt_name"].astype(str)

train_df, val_df = train_test_split(
    df[["text", "label", "prompt_name", "source"]],
    test_size=0.2,
    random_state=42,
    stratify=split_key
)
```

注意：

- `label_prompt` 不是原始 CSV 欄位，若要使用，必須先在程式中動態建立
- 若未來想簡化流程，也可先退回 `stratify=df["label"]`

### 5.4 EDA 項目

EDA 至少應包含：

- Human 與 AI 文本數量分布
- 字數或 token 數分布
- `unique word count`
- `type-token ratio`
- `prompt_name` 分布
- `source` 分布
- 長文本比例與截斷風險

### 5.5 長文本與截斷風險

使用 whitespace token 的近似統計結果：

| 指標 | 數值 |
|------|------|
| whitespace token 平均數 | `383.621` |
| whitespace token 中位數 | `352` |
| whitespace token 最大值 | `1656` |
| 超過 512 whitespace tokens 的文本 | `7,643` |
| 比例 | `17.03%` |

這個 `17.03%` 是長文風險的 proxy，不是 BERT tokenizer 的真實截斷率。由於 `bert-base-cased` / `bert-large-cased` 使用 WordPiece tokenizer，實際被截斷的比例通常會更高，因此報告中應將此點列為限制條件。

### 5.6 Prompt-level Leakage 風險

資料集中有明確的 `prompt_name`。若相同 prompt 的 human / AI 文本同時分布在 train 與 validation，模型可能部分學到 prompt 分布，而非純粹的語言風格差異。這不一定違反作業要求，但應在報告中明確列為 limitation。

---

## 6. 現有程式碼修正清單

### 6.1 `Baseline.py`

目前 `Baseline.py` 可以完成最基本的 baseline 訓練，但還有幾項待修正：

- `train_test_split` 尚未使用 `stratify`
- 目前只保留 `text` 和 `label`，缺少後續可用於 EDA 或聯合分層的欄位
- 只輸出 ROC-AUC，未保存模型、預測分數與圖表

### 6.2 `BERT.py`

目前 `BERT.py` 有幾個會直接導致執行失敗或結果不足的問題：

- 缺少 `import pandas as pd`
- 直接使用未定義的 `X_train`、`X_val`、`y_train`、`y_val`
- `Trainer` 沒有 `compute_metrics`
- `padding="max_length"` 會造成固定 512 padding，浪費顯存
- `output_dir` 與整體 artifact 結構尚未統一

### 6.3 `LocalLLM.py`

目前 `LocalLLM.py` 的主要問題如下：

- 使用未定義的 `X_val`
- 只處理單一樣本，不符合作業要求的 `5-10` 篇 attack 測試
- attack 結果未保存成表格或 CSV
- 沒有批量流程，不利於報告整理

---

## 7. 關鍵程式碼設計

### 7.1 Baseline：TF-IDF + Logistic Regression

```python
vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(train_df["text"])
X_val_tfidf = vectorizer.transform(val_df["text"])

clf = LogisticRegression(
    solver="liblinear"
)

clf.fit(X_train_tfidf, train_df["label"])
probs = clf.predict_proba(X_val_tfidf)[:, 1]
auc = roc_auc_score(val_df["label"], probs)
```

後續可視資源與效果再調整為：

- `ngram_range=(1, 2)`
- `max_features=50000`
- `min_df=2`
- `sublinear_tf=True`

### 7.2 BERT Tokenization 與動態 Padding

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )
```

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

使用動態 padding 的原因是：

- 降低固定補齊到 512 的顯存浪費
- 提升訓練速度
- 更適合長度分布差異大的文本資料

### 7.3 BERT 的評估指標

```python
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }
```

這段是必要項目，因為作業要求用 ROC-AUC 比較模型表現。若沒有 `compute_metrics`，`Trainer.evaluate()` 只會回傳 loss 類指標，不足以完成作業。

### 7.4 BERT 訓練設定建議

建議起始設定如下：

| 項目 | Base | Large |
|------|------|-------|
| Model | `bert-base-cased` | `bert-large-cased` |
| max_length | `512` | `512` |
| per_device_train_batch_size | `16` | `8` |
| gradient_accumulation_steps | `2` | `2` |
| epochs | `3` | `3` |
| learning_rate | `2e-5` | `2e-5` |
| fp16 | `True` | `True` |

若 `bert-large-cased` OOM，優先採取：

1. 降低 batch size
2. 保留 `max_length=512`
3. 增加 gradient accumulation
4. 開啟 gradient checkpointing

### 7.5 Local LLM Attack 流程

```python
attack_samples = val_df[val_df["label"] == 0].sample(n=8, random_state=42)

for text in attack_samples["text"]:
    prompt = (
        "Rewrite the following essay so it sounds natural and student-written:\n\n"
        f"{text}"
    )
    rewritten = generator(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
```

作業主要求是：

- 從 validation set 中選出 `5-10` 篇 human essay
- 用本地 LLM 改寫
- 將改寫後文本送回最佳 BERT detector
- 比較原文與改寫後的 AI probability / logits

### 7.6 Attack Prompt 設計

可準備 2 到 3 種不同的 prompt 做對照：

```python
ATTACK_PROMPTS = [
    "Rewrite the following essay so it sounds more natural and human-written:",
    "Rewrite this essay as if a high school student wrote it in their own words:",
    "Paraphrase the following text while keeping the ideas simple and natural:"
]
```

這有助於比較不同改寫策略對 detector 的影響，但報告主體仍以作業指定的 human-essay rewrite 為主。

---

## 8. 實作步驟

### 第 1 階段：先修正基礎資料流程

1. 新增共用資料模組，統一 CSV 讀取與固定 split
2. 整理 artifact 輸出路徑
3. 讓 `Baseline.py`、`BERT.py`、`LocalLLM.py` 都可獨立執行

### 第 2 階段：完成 Baseline 與 EDA

1. 實作資料統計與圖表輸出
2. 跑 TF-IDF + Logistic Regression
3. 保存 baseline ROC-AUC、預測分數與 baseline 模型

### 第 3 階段：完成 BERT Base

1. 先跑小規模 smoke test
2. 確認 tokenization、Trainer、metrics、保存路徑都正常
3. 執行完整 `bert-base-cased` 訓練
4. 保存 best checkpoint 與 validation metrics

### 第 4 階段：完成 BERT Large

1. 使用同一份 split 與同一套 metrics
2. 調整 batch size 避免 OOM
3. 比較 Base 與 Large 的 ROC-AUC、訓練時間與 VRAM 成本

### 第 5 階段：完成 Local LLM Attack

1. 載入最佳 detector
2. 選取 `5-10` 篇 human essay
3. 以本地 LLM 生成改寫版本
4. 對改寫結果重新進行 detector 推論
5. 保存 attack 結果表與案例文本

### 第 6 階段：完成報告與 GitHub 整理

1. 匯整 TF-IDF、BERT-Base、BERT-Large 比較表
2. 匯出 loss curve、ROC-AUC 或其他圖表
3. 挑選 `1-2` 個 adversarial case 進行分析
4. 整理 README、依賴與執行說明

---

## 9. 測試與驗收標準

### 9.1 Baseline 階段

- 腳本可以獨立執行
- 成功輸出 validation ROC-AUC
- EDA 圖表與資料統計能正確產生

### 9.2 BERT 階段

- `trainer.evaluate()` 能輸出 `roc_auc`
- 模型與 tokenizer 可正確保存與重新載入
- Base 與 Large 使用相同 split 與相同 metrics

### 9.3 Attack 階段

- 至少完成 `5-10` 篇 human essay 的改寫
- 每篇都保留原文、改寫後文本、AI probability 與預測標籤
- 至少能展示 `1-2` 個適合寫入報告的案例

### 9.4 最終報告驗收

報告至少需包含：

- EDA 結果
- TF-IDF、BERT-Base、BERT-Large 的比較表
- scaling trade-off 討論
- adversarial attack 的樣本分析
- 長文本截斷與 prompt leakage 的 limitation 討論

---

## 10. 風險與備案

| 風險 | 說明 | 備案 |
|------|------|------|
| BERT-Large OOM | 24GB VRAM 仍可能因 batch size 過大而溢出 | 降低 batch size、增加 gradient accumulation、開 gradient checkpointing |
| 長文本截斷 | 約 `17.03%` whitespace token 文本已超過 512，真實 tokenizer 截斷比例可能更高 | 在報告中列為 limitation，必要時討論 head+tail 或 sliding window |
| `accelerate` / `bitsandbytes` 相容性 | Windows 上量化套件可能不穩定 | 優先完成 fp16 版本，量化作為進階優化 |
| Llama 3 權限 | 部分模型需要 Hugging Face token | 優先使用 `mistralai/Mistral-7B-Instruct-v0.3` |
| Prompt leakage | 相同 prompt 落在 train/val 可能高估泛化能力 | 在報告中明確討論 limitation |
| 類別不平衡 | 資料比例約 `61:39` | 以 ROC-AUC 為主，必要時觀察 `class_weight='balanced'` 對 baseline 的影響 |

---

## 11. 比較表與輸出模板

### 11.1 模型比較表模板

| Model | ROC-AUC | Accuracy | F1 | Train Time | Peak VRAM | Params | Notes |
|-------|---------|----------|----|------------|-----------|--------|-------|
| TF-IDF + LR |  |  |  |  | N/A |  |  |
| BERT-Base |  |  |  |  |  | 110M |  |
| BERT-Large |  |  |  |  |  | 340M |  |

### 11.2 Attack 紀錄表模板

| essay_id | original_ai_prob | attack_prompt | rewritten_ai_prob | pred_label | fooled_or_not |
|----------|------------------|---------------|-------------------|------------|---------------|
|  |  |  |  |  |  |

---

## 12. 最終交付清單

最終需整理並提交：

- 可執行的 `.py` 腳本
- 必要依賴說明或 `requirements.txt`
- 訓練與 attack 產出的主要圖表與表格
- 模型比較結果
- GitHub 程式碼整理
- PDF 報告

本文件的角色是整個 HW2 的實作藍圖。後續所有程式修正、實驗設計與報告撰寫，皆以此版本為準。
