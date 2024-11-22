import pandas as pd
import glob

# df_utf8 = pd.read_csv("KnowledgeBase\French_Dictionary_Change.csv", encoding="utf-8", nrows=1000)
# print(f"utf8:\n{df_utf8}")
# df_ISO = pd.read_csv("KnowledgeBase\French_Dictionary_Change.csv", encoding="ISO-8859-1", nrows=100, usecols=[0, 1])
# print(f"ISO:\n{df_ISO}")
# df_Big5 = pd.read_csv("KnowledgeBase\French_Dictionary_Change.csv", encoding="Big5", nrows=100, usecols=[0, 1])
# print(f"Big5:\n{df_Big5}")
KnowledgeBases = "KnowledgeBase"
csv_files = glob.glob(f"{KnowledgeBases}/*.csv")
print(csv_files)
for csv_file in csv_files:
    documents = pd.read_csv(csv_file, header=None, encoding="utf-8", nrows=10)
    for index, row_content in documents.iterrows():
        mot = row_content[0]
        mot_definition = row_content[1]
        print(f"index = {index}")
        print(f"mot = {mot}")
        print(f"definition = {mot_definition}")