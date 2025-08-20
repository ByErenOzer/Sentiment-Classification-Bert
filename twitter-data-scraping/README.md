# Twitter Veri Çekme Kodları

Bu klasör Twitter'dan veri çekmek için kullanılan Jupyter Notebook dosyalarını içerir.

## Dosyalar

### 1. saglik_scraping.ipynb
- Sağlık Bakanlığı'na gönderilen tweetleri çeker
- Belirli tarih aralıklarında veri toplama
- `to:@saglikbakanligi` sorgusu kullanır

### 2. teror_scraping.ipynb
- Terör ile ilgili tweetleri çeker
- Çeşitli terör anahtar kelimeleri kullanır
- Türkçe tweetleri filtreler, retweet'leri hariç tutar

### 3. saglik_scraping_2.ipynb
- Sağlık konulu tweetler için alternatif scraping kodu
- Farklı tarih aralıkları ve parametreler

## Kullanım

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas subprocess
   ```

2. Tweet-harvest aracını kullanmak için Node.js gereklidir

3. Her notebook'ta tarih aralıklarını ve token'ları güncelleyin

4. Notebook'ları çalıştırarak veri çekme işlemini başlatın

## Önemli Notlar

- API token'larını güncel tutun
- Rate limit'lere dikkat edin
- Çekilen verileri uygun klasörlerde saklayın
- Veri toplama işlemi zaman alabilir