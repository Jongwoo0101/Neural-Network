import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# windows
# plt.rcParams['font.family'] = 'Malgun Gothic'

# mac
plt.rcParams['font.family'] = 'AppleGothic'

with open("5주차/text.txt", "r", encoding="utf-8") as f:
    text = f.read()

cleaned_text = re.sub(r'[^가-힣\s]', '', text)

tokens = cleaned_text.split()

stopwords = [
    "이","그","저","것","수","등","때","및","들","처럼","까지","에서","으로","에게",
    "이다","하다","되다","했다","합니다","하는","하여","하게","되며","되는","된다","되어","하며",
    "그리고","그러나","하지만","그래서","그러면","따라서","또한","즉","때문에",
    "나","너","우리","저희","그것","이것","저것","사람","누구",
    "중심","부분","경우","모든","각각","다른","같은","어떤","있다","없다","된다"
]

tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

counter = Counter(tokens)
top_20 = counter.most_common(20)
print("상위 20개 단어:", top_20)

wc = WordCloud(
    font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # MacOS
    # Windows: "C:/Windows/Fonts/malgun.ttf"
    background_color="white",
    width=800,
    height=600
)
wc.generate_from_frequencies(dict(top_20))

plt.figure(figsize=(10,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud", fontsize=16)
plt.show()

top_n = 15
most_common = counter.most_common(top_n)
words, freqs = zip(*most_common)

plt.figure(figsize=(10,6))
plt.bar(words, freqs)
plt.title("Top Words Frequency", fontsize=16)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
