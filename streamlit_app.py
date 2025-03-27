import streamlit as st
import pandas as pd
import numpy as np
import faiss
from opencc import OpenCC
import os
import openai

cc = OpenCC('s2twp')  # 簡體轉台灣繁體

# GPT 開關與 API 初始化
enable_gpt = st.sidebar.checkbox("✅ 啟用 GPT 翻譯建議", value=False)
if enable_gpt:
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

def gpt_translate(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個專業的繁體中文翻譯助手，使用台灣用語。"},
                {"role": "user", "content": f"請將以下簡體中文翻譯成台灣常用的繁體中文：\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT 錯誤：{str(e)}"

# ----------- 模擬 embedding 函數 ----------- #
def fake_embed(text):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536).astype("float32")

# ----------- 初始化翻譯記憶庫 ----------- #
data_path = "memory.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    data = {
        "原文（簡體）": [
            "今天是星期五。",
            "他正在學習中文。",
            "我喜歡喝咖啡。",
            "我們一起去公園散步吧。",
            "這部電影非常好看。"
        ],
        "翻譯（繁體）": [
            "今天是星期五。",
            "他正在學習中文。",
            "我喜歡喝咖啡。",
            "我們一起去公園散步吧。",
            "這部電影非常好看。"
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)

# ----------- 建立 FAISS 向量庫 ----------- #
def rebuild_index():
    global index, df
    source_texts = df["原文（簡體）"].tolist()
    embeddings = [fake_embed(text) for text in source_texts]
    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings))

rebuild_index()

# ----------- Streamlit UI ----------- #
st.title("翻譯記憶 繁體翻譯工具（可選 GPT）")
st.write("請輸入一段簡體中文，系統會根據過往翻譯記錄、繁體轉換及 GPT 模型產出建議翻譯：")

user_input = st.text_input("輸入簡體原文")

# 自訂詞彙修正
def apply_custom_dict(text):
    custom_dict = {
        "初始頭像": "預設頭像",
        "賬號": "帳號",
    }
    for k, v in custom_dict.items():
        text = text.replace(k, v)
    return text

if user_input:
    query_vector = fake_embed(user_input)
    D, I = index.search(np.array([query_vector]), k=1)
    best_idx = I[0][0]

    st.subheader("系統找到的最接近翻譯紀錄：")
    suggested_translation = df.iloc[best_idx]['翻譯（繁體）']
    st.write(f"👉 {df.iloc[best_idx]['原文（簡體）']} → {suggested_translation}")

    # 嘗試簡體轉台灣繁體轉換 + 詞彙修正
    converted_text = cc.convert(user_input)
    converted_text = apply_custom_dict(converted_text)

    st.subheader("最終翻譯建議：")
    st.success(converted_text)

    if enable_gpt:
        st.subheader("GPT 翻譯建議：")
        gpt_result = gpt_translate(user_input)
        gpt_result = apply_custom_dict(gpt_result)
        st.info(gpt_result)

    if user_input not in df["原文（簡體）"].values:
        df.loc[len(df)] = [user_input, converted_text]
        df.to_csv(data_path, index=False)
        st.info("✅ 此句已新增至翻譯記憶庫")
        rebuild_index()

# ----------- 批次上傳翻譯功能 ----------- #
st.header("📄 批次上傳簡體文本進行翻譯")
uploaded_file = st.file_uploader("上傳 Excel (.xlsx) 或純文字檔 (.txt)", type=["xlsx", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text_lines = uploaded_file.read().decode("utf-8").splitlines()
        batch_df = pd.DataFrame({"原文（簡體）": [line for line in text_lines if line.strip()]})
    elif uploaded_file.name.endswith(".xlsx"):
        try:
            batch_df = pd.read_excel(uploaded_file)
        except ImportError:
            st.error("請先安裝 openpyxl 套件，或改用 TXT 檔案。")
            st.stop()

        if "原文（簡體）" not in batch_df.columns:
            st.error("Excel 檔案需包含『原文（簡體）』欄位")
            st.stop()

    with st.spinner("正在翻譯..."):
        def memory_lookup(query):
            qv = fake_embed(query)
            _, idxs = index.search(np.array([qv]), k=1)
            base = df.iloc[idxs[0][0]]['翻譯（繁體）']
            converted = cc.convert(query)
            return apply_custom_dict(converted)

        batch_df["翻譯（繁體）"] = batch_df["原文（簡體）"].apply(memory_lookup)

        st.success("翻譯完成！")
        st.dataframe(batch_df)

        csv_download = batch_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載翻譯結果 (CSV)", csv_download, file_name="翻譯結果.csv", mime="text/csv")

# ----------- 上傳字庫功能 ----------- #
st.header("📚 上傳自定義翻譯記憶庫")
dict_file = st.file_uploader("📥 上傳字庫 (需為包含『原文（簡體）』『翻譯（繁體）』欄位的 Excel 檔案)", type="xlsx")
if dict_file:
    try:
        new_dict_df = pd.read_excel(dict_file)
        if "原文（簡體）" in new_dict_df.columns and "翻譯（繁體）" in new_dict_df.columns:
            before = len(df)
            df = pd.concat([df, new_dict_df], ignore_index=True).drop_duplicates(subset="原文（簡體）")
            df.to_csv(data_path, index=False)
            rebuild_index()
            st.success(f"✅ 字庫已成功匯入，共新增 {len(df)-before} 筆資料")
        else:
            st.error("❌ 檔案缺少必要欄位『原文（簡體）』或『翻譯（繁體）』")
    except Exception as e:
        st.error(f"❌ 上傳錯誤：{e}")
