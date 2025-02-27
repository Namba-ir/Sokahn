import unicodedata
import string
import time
from functools import lru_cache
import concurrent.futures
import numpy as np
import csv

COMBINED_TRANS = str.maketrans('', '', string.punctuation + string.whitespace + '@#')
NUMBER_TRANS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')

class TurboNormalizer:
    def __init__(self):
        self.combined_trans = COMBINED_TRANS
        self.number_trans = NUMBER_TRANS
        self.diacritics = set('\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652')
        self.suffixes = {'ها', 'ای', 'تر', 'تری', 'ترین'}
    
    @lru_cache(maxsize=16384)
    def normalize_word(self, word: str) -> str:
        word = unicodedata.normalize('NFKC', word)
        word = ''.join(c for c in word if c not in self.diacritics)
        return word.translate(self.combined_trans).translate(self.number_trans).lower()
    
    def _normalize_chunk(self, sentence: str) -> str:
        sentence = unicodedata.normalize('NFKC', sentence.lower())
        sentence = ''.join(c for c in sentence if c not in self.diacritics)
        sentence = sentence.translate(self.combined_trans).translate(self.number_trans)
        
        # کاهش تکرار حروف
        result = []
        last_char = ''
        count = 0
        for char in sentence:
            if char == last_char:
                count += 1
                if count <= 2:
                    result.append(char)
            else:
                result.append(char)
                last_char = char
                count = 1
        sentence = ''.join(result)
        
        # جدا کردن "می" و "نمی"
        words = sentence.split()
        for i in range(len(words)):
            word = words[i]
            if word.startswith('می') or word.startswith('نمی'):
                prefix = 'می' if word.startswith('می') else 'نمی'
                rest = word[len(prefix):]
                if rest:
                    words[i] = prefix + '‌' + rest
        
        # نیم‌فاصله‌گذاری پسوندها
        output = []
        i = 0
        while i < len(words):
            if i + 1 < len(words) and words[i + 1] in self.suffixes:
                output.append(words[i] + '‌' + words[i + 1])
                i += 2
            else:
                output.append(words[i])
                i += 1
        
        return ' '.join(output)
    
    def normalize_sentence(self, sentence: str) -> str:
        return self._normalize_chunk(sentence)
    
    def normalize_bulk(self, texts: list) -> list:
        if len(texts) < 100:
            return [self._normalize_chunk(t) for t in texts]
        text_array = np.array(texts, dtype=object)
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

# دیتاست نمونه توییتر
twitter_dataset = [
    "سلام چطوری؟ امروز خیلی هوا خوبه!",
    "این فیلم جدید رو دیدم واقعا عالی بود!!!",
    "چرا انقدر ترافیک زیاده؟؟؟ خسته شدم",
    "کاش یه روز بدون استرس داشته باشیم...",
    "دیروز یه کتاب خوندم خیلی جالب بود #کتاب",
    "@ali سلام خوبی؟ کجایی؟",
    "اینترنت باز قطع شده، اعصابم خورد شد!!!",
    "بیا اینو ببین چه باحاله https://t.co/test",
    "امروز حالم خیلی خوبه 😊",
    "نمی‌دونم چرا انقدر خوابم میاد...",
    "هوای بارونی رو خیلی دوست دارم #بارون",
    "این غذا رو درست کردم خیلی خوشمزه شد!",
    "دوباره امتحان دارم دعا کن برام @maryam",
    "چرا قیمتا انقدر گرون شده؟؟",
    "یه چایی داغ الان می‌چسبه...",
    "فوتبال دیشب رو دیدی؟ چه گلی زد!",
    "کاش تعطیلات زودتر برسه #تعطیلات",
    "این آهنگ جدید رو گوش بده فوق‌العادست!",
    "دارم می‌رم خرید، چیزی لازم داری؟",
    "چرا اینقدر کارام عقب افتاده؟؟؟",
    "امروز یه سگ بامزه دیدم خیلی کیوت بود 😍",
    "نمی‌تونم تصمیم بگیرم چی بپوشم...",
    "همه‌چیز گرون شده دیگه نمی‌شه زندگی کرد!",
    "دیشب تا صبح بیدار بودم، الان نابودم",
    "یه ایده باحال به ذهنم رسید #ایده",
    "سلامممممم به همه دوستام!",
    "این عکس رو ببین چقدر قشنگه https://t.co/pic",
    "امروز خیلی کار دارم سرم شلوغه",
    "چرا همیشه کارا رو دقیقه نود می‌کنم؟",
    "کاش یه سفر برم حالم عوض شه...",
    "این جوک رو شنیدی؟ خیلی خنده‌دار بود!",
    "دارم فیلم می‌بینم، مزاحم نشید!",
    "هوا خیلی سرده، پتو پیچ شدم",
    "چقدر دلم برای دوستام تنگ شده...",
    "این آدما چرا انقدر غر می‌زنن؟؟",
    "یه قهوه الان معجزه می‌کنه ☕",
    "امشب مهمون داریم، چی بپزم؟",
    "چرا این کار درست پیش نمی‌ره؟؟؟",
    "دلم یه طبیعت‌گردی می‌خواد #طبیعت",
    "این سریال جدید خیلی قشنگه، ببینیدش!",
    "امروز تولدمه، تبریک یادتون نره 🎉",
    "کاش یه کم آفتاب ببینیم...",
    "چرا انقدر گرسنمه؟؟؟",
    "یه آهنگ قدیمی پیدا کردم خاطراتم زنده شد",
    "سلام به همه، روزتون چطور بود؟",
    "اینترنت سرعتش افتضاحه!",
    "دلم برای دریا تنگ شده #دریا",
    "چرا اینقدر کارام زیاد شده؟",
    "یه چایی با دوستام می‌چسبه الان",
    "فیلم آخر هفته رو انتخاب کردی؟",
    "امروز خیلی خندیدم، روز خوبی بود",
    "چرا اینقدر هوا آلوده‌ست؟؟",
    "کاش می‌شد زمان رو نگه داشت...",
    "یه عکس قشنگ گرفتم بذارم توییتر",
    "دلم یه کیک شکلاتی می‌خواد 🍫",
    "چرا همیشه همه‌چیز گرون می‌شه؟",
    "امشب ماه خیلی قشنگه، دیدیش؟",
    "دارم برنامه‌ریزی می‌کنم برای هفته بعد",
    "چقدر دلم برای خونوادم تنگ شده...",
    "این آهنگ رو تکرار زدم، محشره!",
    "امروز خیلی انرژی دارم 😊",
    "چرا این کارا تمومی نداره؟؟",
    "یه روز آروم می‌خوام بدون دردسر",
    "این بازی جدید رو امتحان کردی؟",
    "سلامممم، چطورید همه؟",
    "چرا انقدر خوابم میاد الان؟",
    "دلم یه سفر جاده‌ای می‌خواد",
    "این غذا رو امتحان کن، عالیه!",
    "امروز یه اشتباه خنده‌دار کردم",
    "چقدر این روزا زود می‌گذره...",
    "کاش یه کم هوا خنک‌تر بشه",
    "یه ایده دارم برات، نظرت چیه؟",
    "چرا اینقدر کارام عقب افتاده؟؟",
    "امشب ستاره‌ها خیلی قشنگن",
    "دارم چایی می‌خورم، تو چی؟",
    "چقدر دلم برای تابستون تنگ شده",
    "این هفته خیلی سخت بود برام",
    "یه فیلم قدیمی پیدا کردم ببینم",
    "سلام به همه، حالم خوبه!",
    "چرا اینقدر همه‌چیز پیچیده‌ست؟",
    "کاش یه روز بدون گوشی باشم...",
    "امروز یه کار باحال کردم",
    "چقدر این آهنگ حس خوبی داره",
    "دلم یه پیاده‌روی می‌خواد",
    "چرا این کار درست نمی‌شه؟؟",
    "یه عکس از غروب گرفتم قشنگ شد",
    "امشب شام چی بخوریم؟",
    "چقدر این روزا خسته‌م...",
    "کاش یه تعطیلی طولانی داشتیم",
    "یه جوک بگم بخندیم؟",
    "دارم به یه سفر فکر می‌کنم",
    "چرا انقدر همه‌چیز گرونه؟",
    "امروز یه خبر خوب شنیدم",
    "چقدر دلم برای برف تنگ شده",
    "یه آهنگ شاد بذار حالم عوض شه",
    "سلاممممم، چطورید دوستام؟",
    "چرا این کارا انقدر طول می‌کشه؟",
    "امشب یه فیلم کمدی می‌بینم",
    "کاش یه روز بدون استرس باشه"
]

if __name__ == "__main__":
    normalizer = TurboNormalizer()
    
    tests = {
        "تک کلمه ساده": ("TEST!", normalizer.normalize_word),
        "جمله استاندارد": ("این یک متن TEST با علائم !@# سجاوندی و   فاصله‌های اضافه است!!!", normalizer.normalize_sentence),
        "پردازش گروهی": (twitter_dataset, normalizer.normalize_bulk),
        "توییت نمونه": ("سلام @ali چطوری؟ #جالب اینو ببینممم https://t.co/test 😂 ۳.۱۴", normalizer.normalize_sentence)
    }
    
    # ذخیره نتایج در CSV
    with open('normalized_tweets.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Text', 'Normalized Text', 'Execution Time (s)'])  # سرستون‌ها
        
        for name, (data, func) in tests.items():
            result, elapsed = precision_timer(func)(data)
            print(f"✅ تست {name}")
            print(f"⏱ زمان اجرا: {elapsed:.10f} ثانیه")
            print(f"📊 حجم داده: {len(data) if isinstance(data, list) else len(data.split())}")
            
            if isinstance(result, str):  # برای تک متن‌ها
                writer.writerow([data, result, elapsed])
                print(f"🎯 نمونه خروجی: {result[:75] + '...' if len(result) > 75 else result}")
            else:  # برای لیست‌ها (مثل پردازش گروهی)
                for orig, norm in zip(data, result):
                    writer.writerow([orig, norm, elapsed / len(data)])  # زمان تقریبی هر توییت
                print(f"🎯 نمونه خروجی: {result[0][:75] + '...' if len(result[0]) > 75 else result[0]}")
            print("-" * 80)
    
    print("نتایج در فایل 'normalized_tweets.csv' ذخیره شد!")