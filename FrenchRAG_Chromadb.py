import glob
import chromadb
import pandas as pd
import ollama
import gradio as gr
class FrenchRAG():
    _instance = None
    _initialized = False  # 標記是否已初始化
    collection = ""
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FrenchRAG, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        VectorDB = "VectorDB"
        client = chromadb.PersistentClient(path=VectorDB)
        self.collection = client.get_or_create_collection(name="FrenchWord")
        current_count = self.collection.count()
        print(f"Current vector count: {current_count}")
        # 如果資料數量為 0，則認為尚未加入資料
        if current_count == 0:
            print("No vectors found in the collection. Creating RAG...")
            self.createRAG()
        else:
            print(f"Collection already contains {current_count} vectors.")

    def createRAG(self):
        if not self._initialized:  # 檢查是否已初始化
            KnowledgeBases = "KnowledgeBase"
            csv_files = glob.glob(f"{KnowledgeBases}/*.csv")
            print(csv_files)
            batch_size = 50
            ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
            for csv_file in csv_files:
                documents = pd.read_csv(csv_file, header=None, encoding="utf-8")
                header = documents.iloc[0]  # 第一行作為標頭
                documents = documents[1:]  # 移除第一行
                for index, row_content in documents.iterrows():
                    print(f"index = {index}")
                    mot = row_content[0]
                    mot_definition = row_content[1]
                    response = ollama.embeddings(model="mxbai-embed-large", prompt=mot_definition)
                    ids_batch.append(f"ID{index}")
                    # ids_batch.append(f"{csv_file}_ID{index}")
                    embeddings_batch.append(response["embedding"])
                    documents_batch.append(mot + ":" + mot_definition)
                    metadatas_batch.append({"word":mot})
                    print(f"ids_batchs len = {len(ids_batch)}")
                    if (len(ids_batch)%100) == batch_size:
                        self.collection.add(ids=ids_batch, embeddings=embeddings_batch, documents=documents_batch, metadatas=metadatas_batch)
                        ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []  # 清空批次
                print(f"Knowledge data {csv_file} finished turn into vector!")            
            FrenchRAG._initialized = True  # 標記為已初始化
        
class Chatllama():
    def __init__(self):
        self.LLM = "llama3.2"
        self.embeddingModel = "mxbai-embed-large"
        frenchRAG = FrenchRAG()
        self.colletion = frenchRAG.collection
        self.prePrompt = """
You will act as an expert in French, skilled at analyzing the semantics, grammatical properties, and contextual applications of French words. Based on a given word, you will identify 10 synonyms or near-synonyms with the same grammatical category. These will be classified and explained according to their emotional intensity and formality.

Please adhere to the following rules for your response:

Provide two paragraphs in total:
First paragraph: List the "replaced word" and the emotional intensity and formality of the 10 "replacement words."
Second paragraph: Provide a brief explanation of the specific meaning and usage context for each "replacement word."
The grammatical category of the replaced word and replacement words must be identical (e.g., all nouns or all adjectives).
For each "replacement word," include:
Emotional intensity: Rated from 1 to 5 (1 being the lowest, 5 being the highest), indicating the strength of emotional expression.
Formality: Rated from 1 to 5 (1 being the lowest, 5 being the highest), indicating whether the word is suited for informal or formal contexts.
The emotional intensity and formality of the replaced word are fixed at 3 (medium level).
Ensure diversity by covering all intensity and formality scores (i.e., 1 through 5).
Only provide synonyms or near-synonyms; avoid antonyms.
Explanations should be concise and highlight differences in emotional intensity and formality in context.
Arrange the replacement words from lowest to highest emotional intensity.
Do not repeat replacement words.
Ensure the order of explanation matches the order of the listed replacement words.

Here is the example format:
Example Question:
Please list 10 synonyms for "colère."

Example Answer:
Word/Emotional Intensity/Formality
Replaced word: colère 3 3

1. mécontentement 1 2
2. agacement 2 3
3. indignation 3 4
4. courroux 4 5
5. rage 5 1
6. énervement 1 1
7. furie 3 2
8. emportement 1 3
9. irritation 2 4
10. fureur 4 5

Explanation:
1. mécontentement: Displeasure or dissatisfaction, typically used to express mild disappointment or discontent with a situation. Informal tone.
2. agacement: A mild emotional reaction indicating annoyance. Used in both spoken and written contexts.
3. indignation: Outrage or strong discontent, often in response to unfairness or immorality. Suited for formal settings.
4. courroux: Lofty anger, intense and dignified. Very formal, often found in literature or written works.
5. rage: Extreme anger, the most intense form. Common in informal and direct contexts, with a rough and raw tone.
6. énervement: Irritation or nervousness, low emotional intensity, carrying a sense of discomfort. Used informally in daily conversation.
7. furie: Fury, anger with a slightly exaggerated tone. Informal and often dramatic in spoken French.
8. emportement: A brief loss of composure, which could manifest as anger or impatience. Slightly formal, used to describe temporary emotional lapses.
9. irritation: Mild irritation or provocation, usually triggered by minor issues or continuous disturbances. Formal tone, often used to describe controlled emotions.
10. fureur: Wrath or extreme passion, characterized by a destructive and intense nature. Commonly found in literary works or dramatic descriptions.

Example Question:
Please list 10 synonyms for "peur."

Example Answer:
Word/Emotional Intensity/Formality
Replaced word: peur 3 3

1. inquiétude 1 2
2. angoisse 2 3
3. crainte 3 3
4. horreur 4 1
5. effroi 5 5
6. trac 2 1
7. panique 1 2
8. épouvante 4 3
9. appréhension 2 4
10. affres 5 5

Explanation:
1. inquiétude: Concern or unease, often conveying mild worry about something. Common in informal contexts and used to describe limited anxiety.
2. angoisse: Anxiety or deep worry, usually related to the future or a particular event. More intense than "inquiétude," suited for moderately formal contexts.
3. crainte: Fear, a sense of unease about potential threats or risks. Moderate intensity, suitable for both informal and formal settings.
4. horreur: Terror, extreme fear, or aversion to something. Very intense and dramatic, often used in informal scenarios.
5. effroi: Fear or shock, very strong emotional reaction to a major event or disaster. Highly formal, typically found in written or professional settings.
6. trac: Stage fright, anxiety related to public performance. Usually temporary and linked to social situations. Commonly informal.
7. panique: Panic, a strong and sudden emotional response to danger or chaos. Casual and colloquial in use.
8. épouvante: Dread or shock, used to describe a severe emotional reaction to an event. Applicable in both spoken and written forms.
9. appréhension: Apprehension, typically indicating worry about future uncertainty. Medium intensity, suitable for formal contexts when describing rational concerns.
10. affres: Extreme anguish or fear, signifying profound psychological torment or extreme dread. Formal and literary, often used in elevated discourse.

Please strictly follow the format of the two examples above to respond.
"""
#         self.prePrompt = """
# 你將扮演一位精通法語的專家，擅長分析法語單詞的語意、詞性及應用情境，並能根據提供的單詞找出10個詞性相同的同義詞或近義詞，並按照情緒強度和正式性進行分類與解釋。

# 請按照以下規則回應：
# 1. 總共分為 **兩個回答段落**：
#     - 第一段：列出「被替換單詞」及10個「替換單詞」的情緒強度與正式性。
#     - 第二段：逐一解釋每個「替換單詞」的具體含義與使用情境。
# 2. **被替換單詞與替換單詞的詞性需完全一致**（如都是名詞或形容詞）。
# 3. 每個「替換單詞」需提供以下資訊：
#     - **情緒強度**：從1到5分（1最低，5最高），表示該單詞情感表達的強烈程度。
#     - **正式性**：從1到5分（1最低，5最高），表示該單詞適用於非正式還是正式語境。
# 4. 被替換單詞的情緒強度與正式性固定為3（中等）。
# 5. 必須包含情緒強度和正式性範圍內的每個分數（即1到5），確保多樣性。
# 6. 僅提供同義詞或近義詞，避免反義詞。
# 7. 解釋應簡明扼要，突出單詞的情感強度與正式性在語境中的差異。
# 8. 替換單詞應該由情緒低往高排列
# 9. 替換單詞不可重複
# 10. 替換單詞語解釋單詞的次序應該相同

# 以下是範例格式：
# ---
# Example Question:  
# 請幫我列出「colère」的10個同義詞。

# Example Answer:  
# **單詞/情緒強度/正式性**  
# 被替換單詞：colère 3 3  
# 1. mécontentement 1 2  
# 2. agacement 2 3  
# 3. indignation 3 4  
# 4. courroux 4 5  
# 5. rage 5 1  
# 6. énervement 1 1  
# 7. furie 3 2  
# 8. emportement 1 3  
# 9. irritation 2 4  
# 10. fureur 4 5  

# **解釋：**  
# 1. mécontentement：不滿或不快，通常用於表達輕微的失望或對情況不滿意的情緒。較口語。
# 2. agacement：一種輕微的情緒波動，帶有煩擾的意味。口語、書面接可。
# 3. indignation：憤慨，對不公平或不道德行為的強烈不滿。適合正式語境，如對不公平現象的憤怒表達。
# 4. courroux：怒火，強烈而高尚的憤怒。非常正式，通常用於書面或文學作品中。
# 5. Rage：狂怒，極端的憤怒，情緒最強烈。常見於口語和非正式情境，語氣粗獷直接。
# 6. énervement：煩躁或緊張，情緒低，帶有不安的感覺。非正式的，用於日常對話中表達煩悶情緒。
# 7. furie：怒氣，略帶誇張的憤怒情緒。非正式，口語中常見，具有戲劇化的語氣。
# 8. emportement：一時的情緒失控，可能是憤怒或急躁的衝動。稍正式的語境，用於描述短暫的情緒失控。
# 9. irritation：輕微的惱怒或刺激，通常因小事或連續的干擾引發。偏正式，常用於描述有理性控制的情緒。
# 10. fureur：怒火，極度激烈的情緒，帶有毀滅性和劇烈的特質。常見於文學作品或戲劇性的描述。

# Example Question:
# 請幫我列出「peur」的10個同義詞。

# Example Answer:
# **單詞/情緒強度/正式性**  
# 被替換單詞：peur 3 3
# 1. inquiétude 1 2
# 2. angoisse 2 3
# 3. crainte 3 3
# 4. horreur 4 1
# 5. effroi 5 5
# 6. trac 2 1
# 7. panique 1 2
# 8. épouvante 4 3
# 9. appréhension 2 4
# 10. affres 5 5

# **解釋：** 
# 1. inquiétude：擔心或不安，通常表達對某事的輕微擔憂。較口語，常用於日常對話中描述小範圍的焦慮。
# 2. angoisse：焦慮，對未來或某事的極度擔心，通常伴隨生理反應如心跳加速等。比 inquiétude 更強烈，情緒波動較大，適合在稍正式的語境下使用。
# 3. crainte：害怕，對潛在威脅或風險的預感，情緒中等偏高，常用於描述對未來事件或情境的不安。適用於口語與正式語境。
# 4. horreur：恐怖，極端的害怕或對某物極度反感，情緒強烈，帶有戲劇化的色彩。常見於非正式語境，尤其是用來形容令人恐懼的事物或經歷。
# 5. effroi：恐懼，震驚，情緒非常強烈，通常用於描述對重大事件或災難的反應。屬於非常正式的語言，常見於書面語和正式場合。
# 6. trac：怯場，對公開表現的焦慮或恐懼，通常是短暫的，與社交場合或演講有關。常用於非正式語境。
# 7. panique：恐慌，強烈的驚慌情緒，常在危險或混亂中爆發，情緒波動極大。口語化，通常描述突如其來的驚慌。
# 8. épouvante：驚恐，通常用來形容對某事件的劇烈驚駭。口語或書面皆可。
# 9. appréhension：擔憂，通常指對未來不確定性的預感，情緒波動中等，較正式，適用於描述對未來事件的理性擔心。
# 10. affres：極度的痛苦或恐懼，描述一種深刻的心理折磨或極端的恐懼。正式且文學性強，常用於書面語或高級語境中。

# 請嚴格根據以上兩個範例格式回答。
# """
#         self.example = """
# 請根據以下範例格式回應
# Example Question 1：請幫我列出colère的10個同義詞
# Example Answer 1：
# 單詞/情緒強度/正式性
# 被替換單詞：colère 3 3
# 單詞1：mécontentement 1 2
# 單詞2：agacement 2 3
# 單詞3：indignation 3 4
# 單詞4：courroux 4 5 
# 單詞5：Rage 5 1 
# 單詞6：énervement 1 1
# 單詞7：furie 3 2
# 單詞8：emportement 1 3
# 單詞9：Irritation 2 4
# 單詞10：fureur 4 5

# 解釋：
# mécontentement：不滿或不快，通常用於表達輕微的失望或對情況不滿意的情緒。較口語。
# agacement：一種輕微的情緒波動，帶有煩擾的意味。口語、書面接可。
# indignation：憤慨，對不公平或不道德行為的強烈不滿。適合正式語境，如對不公平現象的憤怒表達。
# courroux：怒火，強烈而高尚的憤怒。非常正式，通常用於書面或文學作品中。
# Rage：狂怒，極端的憤怒，情緒最強烈。常見於口語和非正式情境，語氣粗獷直接。
# énervement：煩躁或緊張，情緒低，帶有不安的感覺。非正式的，用於日常對話中表達煩悶情緒。
# furie：怒氣，略帶誇張的憤怒情緒。非正式，口語中常見，具有戲劇化的語氣。
# emportement：一時的情緒失控，可能是憤怒或急躁的衝動。稍正式的語境，用於描述短暫的情緒失控。
# irritation：輕微的惱怒或刺激，通常因小事或連續的干擾引發。偏正式，常用於描述有理性控制的情緒。
# fureur：怒火，極度激烈的情緒，帶有毀滅性和劇烈的特質。常見於文學作品或戲劇性的描述。

# Example Question 2：請幫我列出peur的10個同義詞
# Example Answer 2：
# 單詞/情緒強度/正式性
# 被替換單詞：peur 3 3
# 單詞1：inquiétude 1 2
# 單詞2：angoisse 2 3
# 單詞3：crainte 3 3
# 單詞4：horreur 4 1 
# 單詞5：effroi 5 5
# 單詞6：trac 2 1
# 單詞7：panique 1 2
# 單詞8：épouvante 4 3
# 單詞9：appréhension 2 4
# 單詞10：Affres 5 5

# 解釋：
# inquiétude：擔心或不安，通常表達對某事的輕微擔憂。較口語，常用於日常對話中描述小範圍的焦慮。
# angoisse：焦慮，對未來或某事的極度擔心，通常伴隨生理反應如心跳加速等。比 inquiétude 更強烈，情緒波動較大，適合在稍正式的語境下使用。
# crainte：害怕，對潛在威脅或風險的預感，情緒中等偏高，常用於描述對未來事件或情境的不安。適用於口語與正式語境。
# horreur：恐怖，極端的害怕或對某物極度反感，情緒強烈，帶有戲劇化的色彩。常見於非正式語境，尤其是用來形容令人恐懼的事物或經歷。
# effroi：恐懼，震驚，情緒非常強烈，通常用於描述對重大事件或災難的反應。屬於非常正式的語言，常見於書面語和正式場合。
# trac：怯場，對公開表現的焦慮或恐懼，通常是短暫的，與社交場合或演講有關。常用於非正式語境。
# panique：恐慌，強烈的驚慌情緒，常在危險或混亂中爆發，情緒波動極大。口語化，通常描述突如其來的驚慌。
# épouvante：驚恐，通常用來形容對某事件的劇烈驚駭。口語或書面皆可。
# appréhension：擔憂，通常指對未來不確定性的預感，情緒波動中等，較正式，適用於描述對未來事件的理性擔心。
# affres：極度的痛苦或恐懼，描述一種深刻的心理折磨或極端的恐懼。正式且文學性強，常用於書面語或高級語境中。
# """
    def handle_user_input(self, user_input):
        response = ollama.embeddings(prompt=user_input, model=self.embeddingModel)
        results = self.colletion.query(query_embeddings=[response["embedding"]], n_results=10)
        data = results['documents'][0]
        datatext = ""
        for i, item in enumerate(data):
            # 拆解每個資料條目
            key_value = item.split(":")
            key = key_value[0].strip()
            # 取得方括號內的項目並移除首尾空格與引號
            values = eval(key_value[1].strip())
            datatext += f"{i+1}.{key}:{values}\n"
        print(f"datatext = {datatext}")
        full_prompt = self.prePrompt + f"請善用以下資料：{datatext}\n並請使用法語書寫單詞，用繁體中文 zh-TW 回答除了被法語單詞之外的所有文字。請回答：{user_input}"
        output = ollama.generate(
            model = self.LLM,
            prompt = full_prompt,
        )
        print(output["response"])
        return output["response"]

class gradioUI():
    def __init__(self):
        chatllama = Chatllama()
        with gr.Blocks() as demo:
            user_input = gr.Textbox(label="輸入你的文字", placeholder="請輸入文字...")
            output_text = gr.Textbox(label="回應文字")
            submit_btn = gr.Button("送出")
            
            submit_btn.click(
                fn=chatllama.handle_user_input,
                inputs=user_input,
                outputs=output_text
            )
        demo.launch()

if __name__ == "__main__":
    webUI = gradioUI()