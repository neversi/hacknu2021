import math, json, colorama
import spacy
import time
colorama.init()

nlp = spacy.load("en_core_web_lg")
N = 142570                      # Amount of Documents
AVGDL = 263                     # Average Document Length
content_by_docId = {}           # 1631: (290, 'The Presidential elections are .... ')
freq_by_word_and_docId = {}     # 'dara': {123: 21, 124: 14}, 'abdr': {}
docCount_by_word = {}           # 'dara': 123
K = 2                       # Advanced optimization
B = 0.75
FIRST_CHARS = 600
# ---------------------- Loading ----------------------------

with open('result2.json') as file:
    content_by_docId = json.load(file)
    
with open('result4.json') as file:
    freq_by_word_and_docId = json.load(file)

with open('result1.json') as file:
    docCount_by_word = json.load(file)

# --------------------- BM25 Algorithm


def bm25(docId, query):
    score = 0
    for word in query:
        if word not in freq_by_word_and_docId or docId not in freq_by_word_and_docId[word]:
            continue
        IDF = inverse_document_frequency(word)
        freq = freq_by_word_and_docId[word][docId]
        docSize = content_by_docId[docId][0]
        score += IDF * ((freq * (K + 1)) / (freq + K * (1 - B + B * docSize / AVGDL)))
    return score


def inverse_document_frequency(word):
    return math.log(N / docCount_by_word[word], 2)

# ------------------------- Queries -----------------------------


print("\u001b[34m[*] We are ready for querying\u001b[0m")

def simplify(query: str = ""):
    doc = nlp(query)
    list_str = [token.lemma_.lower().strip() for token in doc]
    list_ent = [ent.text.lower().strip() for ent in doc.ents]
    return list_str.extend(list_ent)

while True:
    # input
    
    query = input("\u001b[36;1m")
    print("\u001b[0m")
    query = simplify(query)
    # print(query)
    ranking = []

    # calculation by BM25
    start = time.perf_counter()
    for docId in content_by_docId.keys():
        ranking.append((bm25(docId, query), docId))

    ranking.sort(reverse=True)
    print("Found in ", round(time.perf_counter() - start, 3))
    # output
    # print(len(ranking))
    for i in range(5):
        doc = ranking[i]
        print('#', i + 1)
        print("\u001b[32;1mScore:\u001b[0m ", doc[0], "\t \u001b[36;1mid:\u001b[0m ", doc[1])
        print("\u001b[35;1mContent:\u001b[0m ", content_by_docId[doc[1]][1][:600])
        # print('###############################')

