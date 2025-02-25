import unicodedata
import string
import time
from functools import lru_cache
import concurrent.futures
import numpy as np

# جداول از پیش‌ساخته با دسترسی سریع‌تر
PUNCTUATION_TRANS = str.maketrans('', '', string.punctuation)
WHITESPACE_TRANS = str.maketrans('', '', string.whitespace)

class TurboNormalizer:
    def __init__(self):
        # تبدیل جداول به فرم قابل استفاده مستقیم
        self.combined_trans = str.maketrans('', '', string.punctuation + string.whitespace)
    
    @lru_cache(maxsize=32768)  # کاهش اندازه کش برای سرعت بیشتر در دسترسی
    def normalize_word(self, word: str) -> str:
        return unicodedata.normalize('NFKC', word).translate(self.combined_trans).lower()
    
    def _normalize_chunk(self, sentence: str) -> str:
        """پردازش سریع یک تکه متن"""
        return ' '.join(
            unicodedata.normalize('NFKC', sentence.lower())
            .translate(self.combined_trans)
            .split()
        )
    
    def normalize_sentence(self, sentence: str) -> str:
        return self._normalize_chunk(sentence)
    
    def normalize_bulk(self, texts: list) -> list:
        # استفاده از پردازش موازی برای داده‌های بزرگ
        if len(texts) < 100:  # برای داده‌های کوچک از روش ساده استفاده کن
            return [self._normalize_chunk(t) for t in texts]
        
        # تبدیل به آرایه NumPy برای مدیریت بهتر حافظه
        text_array = np.array(texts, dtype=str)
        chunk_size = max(1, len(texts) // (4 * concurrent.futures.ThreadPoolExecutor()._max_workers))
        
        # پردازش موازی
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self._normalize_chunk,
                text_array
            ))
        return results

def precision_timer(func):
    def wrapper(*args, **kwargs):
        # گرم کردن بدون حلقه اضافی
        func(*args, **kwargs)
        
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        return result, elapsed
    return wrapper

if __name__ == "__main__":
    normalizer = TurboNormalizer()
    sample_text = "این یک متن TEST با علائم !@# سجاوندی و   فاصله‌های اضافه است!"
    large_dataset = [sample_text * 100 for _ in range(1000)]
    
    tests = {
        "تک کلمه ساده": ("TEST!", normalizer.normalize_word),
        "جمله استاندارد": (sample_text, normalizer.normalize_sentence),
        "پردازش گروهی": (large_dataset, normalizer.normalize_bulk)
    }
    
    for name, (data, func) in tests.items():
        result, elapsed = precision_timer(func)(data)
        print(f"✅ تست {name}")
        print(f"⏱ زمان اجرا: {elapsed:.10f} ثانیه")
        print(f"📊 حجم داده: {len(data) if isinstance(data, list) else len(data.split())}")
        print(f"🎯 نمونه خروجی: {result[:75] + '...' if isinstance(result, str) else result[0][:75] + '...'}")
        print("-" * 80)