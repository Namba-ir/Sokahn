import unicodedata
import string
import time
from functools import lru_cache
import concurrent.futures
import numpy as np
import re

# جداول از پیش‌ساخته
COMBINED_TRANS = str.maketrans('', '', string.punctuation + string.whitespace + '@#')
NUMBER_TRANS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')

class TurboNormalizer:
    def __init__(self):
        self.combined_trans = COMBINED_TRANS
        self.number_trans = NUMBER_TRANS
        # الگوهای اعراب و کاراکترهای خاص
        self.diacritics_pattern = r'[\u064b-\u0652]'  # FATHATAN تا SUKUN
        self.special_chars_pattern = r'[\u0600-\u06FF\ufb50-\ufdff\ufe70-\ufeff]'
        self.repeat_pattern = r'(.)\1{2,}'  # برای کاهش تکرار حروف
        
    @lru_cache(maxsize=32768)
    def normalize_word(self, word: str) -> str:
        word = unicodedata.normalize('NFKC', word)
        word = re.sub(self.diacritics_pattern, '', word)  # حذف اعراب
        word = word.translate(self.number_trans)  # تبدیل اعداد
        return word.translate(self.combined_trans).lower()
    
    def _normalize_chunk(self, sentence: str) -> str:
        sentence = unicodedata.normalize('NFKC', sentence.lower())
        sentence = re.sub(self.diacritics_pattern, '', sentence)  # حذف اعراب
        sentence = sentence.translate(self.number_trans)  # تبدیل اعداد
        
        # کاهش تکرار حروف (بیش از ۲ به ۲)
        sentence = re.sub(self.repeat_pattern, r'\1\1', sentence)
        
        # جایگزینی سبک فارسی
        sentence = sentence.replace('"', '«').replace('"', '»').replace('...', '…').replace('.', '٫')
        
        # جدا کردن "می" و "نمی" با نیم‌فاصله
        sentence = re.sub(r'\b(ن?می)([^ ]+)', r'\1‌\2', sentence)
        
        # حذف کاراکترهای خاص و فاصله‌گذاری
        sentence = re.sub(self.special_chars_pattern, '', sentence)
        words = sentence.split()
        # اضافه کردن نیم‌فاصله برای پسوندهای رایج
        for i in range(len(words)-1, 0, -1):
            if words[i] in {'ها', 'ای', 'تر', 'تری', 'ترین'}:
                words[i-1] = words[i-1] + '‌' + words[i]
                words.pop(i)
        
        return ' '.join(words)
    
    def normalize_sentence(self, sentence: str) -> str:
        return self._normalize_chunk(sentence)
    
    def normalize_bulk(self, texts: list) -> list:
        if len(texts) < 100:
            return [self._normalize_chunk(t) for t in texts]
        
        text_array = np.array(texts, dtype=str)
        chunk_size = max(1, len(texts) // (4 * concurrent.futures.ThreadPoolExecutor()._max_workers))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._normalize_chunk, text_array))
        return results

def precision_timer(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        return result, elapsed
    return wrapper

if __name__ == "__main__":
    normalizer = TurboNormalizer()
    sample_text = "این یک متن TEST با علائم !@# سجاوندی و   فاصله‌های اضافه است!!!"
    large_dataset = [sample_text * 100 for _ in range(1000)]
    twitter_sample = "سلام @ali چطوری؟ #جالب اینو ببینممم https://t.co/test 😂 ۳.۱۴"
    
    tests = {
        "تک کلمه ساده": ("TEST!", normalizer.normalize_word),
        "جمله استاندارد": (sample_text, normalizer.normalize_sentence),
        "پردازش گروهی": (large_dataset, normalizer.normalize_bulk),
        "توییت نمونه": (twitter_sample, normalizer.normalize_sentence)
    }
    
    for name, (data, func) in tests.items():
        result, elapsed = precision_timer(func)(data)
        print(f"✅ تست {name}")
        print(f"⏱ زمان اجرا: {elapsed:.10f} ثانیه")
        print(f"📊 حجم داده: {len(data) if isinstance(data, list) else len(data.split())}")
        print(f"🎯 نمونه خروجی: {result[:75] + '...' if isinstance(result, str) else result[0][:75] + '...'}")
        print("-" * 80)