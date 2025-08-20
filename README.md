###  Twitter Sentiment Analysis
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
