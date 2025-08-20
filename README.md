# Turkish NLP Projects

Turkish natural language processing projects including sentiment analysis and SMS spam detection.

## Projects

### 1. SMS Spam Detection
Turkish SMS spam detection using BERT model for binary classification (Ham/Spam).

#### Performance Results
- **Final Test Accuracy**: 97.69%
- **Final Test F1-Score**: 97.69%
- **Best Validation F1-Score**: 99.34% (Epoch 2)
- **Best Validation Accuracy**: 99.34%

#### Training History
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|------------|-----------|----------|---------|--------|
| 1     | 0.1813     | 92.99%    | 0.0443   | 99.01%  | 99.01% |
| 2     | 0.0310     | 99.22%    | 0.0205   | 99.34%  | 99.34% |

#### Files
- `classification_and_preprocess_code/sms_spam_detection.py` - Main SMS spam detection model
- `classification_and_preprocess_code/sms-tespiti.ipynb` - SMS detection notebook

#### Requirements
```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm
```

#### Usage
```python
python classification_and_preprocess_code/sms_spam_detection.py
```

#### Model
Using `dbmdz/bert-base-turkish-uncased` for Turkish SMS classification into 2 categories:
- Ham (Normal)
- Spam (İstenmeyen)

### 2. Twitter Sentiment Analysis
Turkish sentiment analysis using BERT model for Twitter data classification.

#### Performance
- **Accuracy**: 86%
- **F1 Score**: 85%

#### Files
- `veri_etiketleme/bert_sentiment_analysis.py` - Main sentiment analysis model
- `veri_etiketleme/yenipreprocess (1).ipynb` - Data preprocessing notebook

#### Model
Using `dbmdz/bert-base-turkish-uncased` for Turkish text classification into 3 sentiment categories:
- Negative
- Neutral  
- Positive

#### Usage
```python
python veri_etiketleme/bert_sentiment_analysis.py
```