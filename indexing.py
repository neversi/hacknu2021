import csv
import sys
import spacy
import json
import time
import math
from spacy.tokens import Doc

csv.field_size_limit(sys.maxsize)
spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
Doc.set_extension("did", default=0)

files = ["articles1.csv", "articles2.csv", "articles3.csv"]

N = 142570
# N = 10000
articles = []

start = time.process_time()

for file in files:
    with open("./archive/" + file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_index = 0
        
        for row in csv_reader:
                line_index += 1
                if line_index == 1:
                        continue
                if line_index % 200 == 0:
                        print('stage #', file[-5], ' ', round(line_index/1500, 4), '%\t', line_index, sep='')
                if line_index > N:
                        break
                articles.append((row[-1], {"did": int(row[1])}))

end = time.process_time()

print ("Read time: ", end - start)

dict1 = {}
dict2 = {}
dict3 = {}
dict4 = {}

start = time.perf_counter()
doc_index = 0

# for article in articles:
#         doc = nlp(article[0])
#         meta = article[1]
#         dict2[meta["did"]] = (len(doc), doc.text)
#         local_dict = dict()
#         for token in doc:
#                 if token.is_stop or token.is_punct:
#                         continue
#                 word = token.lemma_.lower().strip()
#                 if word not in local_dict:
#                         local_dict[word] = 1
#                 else:
#                         local_dict[word] += 1
        
#         for key, val in local_dict.items():
#                 if key not in dict1:
#                         dict1[key] = 1
#                 else:
#                         dict1[key] += 1
#                 if key not in dict4:
#                         dict4[key] = {}
#                 dict4[key][meta["did"]] = val
#         doc_index += 1
#         if doc_index % 10 == 0:
#                 print(doc_index)

for doc, meta in nlp.pipe(articles, as_tuples=True, batch_size=15):
        dict2[meta["did"]] = (len(doc), doc.text)
        local_dict = dict()
        for token in doc:
                if token.is_stop or token.is_punct:
                        continue
                word = token.lemma_.lower().strip()
                if word not in local_dict:
                        local_dict[word] = 1
                else:
                        local_dict[word] += 1
        
        for key, val in local_dict.items():
                if key not in dict1:
                        dict1[key] = 1
                else:
                        dict1[key] += 1
                if key not in dict4:
                        dict4[key] = {}
                dict4[key][meta["did"]] = val
        doc_index += 1
        if doc_index % 10 == 0:
                print(doc_index)
end = time.perf_counter()

print("Parse time: ", end - start)

with open("result1.json", "w") as out:
        json.dump(dict1, out)

with open("result2.json", "w") as out:
        json.dump(dict2, out)

with open("result4.json", "w") as out:
        json.dump(dict4, out)
        