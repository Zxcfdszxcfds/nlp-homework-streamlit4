import streamlit as st
import nltk
from nltk import ngrams
from nltk.lm import Laplace, MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import torch
import torch.nn as nn
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="语言模型训练与对比分析平台",
    page_icon="📚",
    layout="wide"
)

# ---------------------- 预下载 NLTK 数据 ----------------------
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')

download_nltk_data()

# ---------------------- 模块1：n元语言模型与数据平滑（修复版：不用reuters语料） ----------------------
# 内置模拟语料，避免reuters报错
CORPUS = [
    ["the", "company", "reported", "a", "profit"],
    ["the", "stock", "market", "is", "rising"],
    ["apple", "released", "a", "new", "product"],
    ["the", "economy", "is", "growing"],
    ["tech", "companies", "are", "hiring"],
    ["the", "ceo", "announced", "new", "plans"],
    ["sales", "increased", "by", "ten", "percent"],
    ["the", "quarterly", "results", "were", "strong"]
]

def train_ngram_model(n=3, use_smoothing=False):
    train_data, padded_vocab = padded_everygram_pipeline(n, CORPUS)
    if use_smoothing:
        model = Laplace(n)
    else:
        model = MLE(n)
    model.fit(train_data, padded_vocab)
    return model

def get_ngram_prob(model, sentence, n=3):
    tokens = nltk.word_tokenize(sentence.lower())
    if len(tokens) < n:
        tokens = ['<s>']*(n-1) + tokens + ['</s>']
    else:
        tokens = ['<s>']*(n-1) + tokens + ['</s>']
    ngrams_list = list(ngrams(tokens, n))
    prob = 1.0
    for gram in ngrams_list:
        prob *= model.score(gram[-1], gram[:-1])
    return prob

# ---------------------- 模块2：从零训练RNN语言模型 ----------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def train_char_rnn(text, hidden_size, epochs, lr):
    chars = sorted(list(set(text)))
    char_to_idx = {c:i for i,c in enumerate(chars)}
    idx_to_char = {i:c for i,c in enumerate(chars)}
    vocab_size = len(chars)
    
    seq_length = 10
    data = [char_to_idx[c] for c in text]
    x, y = [], []
    for i in range(0, len(data)-seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+1:i+seq_length+1])
    x = torch.tensor(x)
    y = torch.tensor(y)
    
    model = CharRNN(vocab_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    model.train()
    for epoch in range(epochs):
        hidden = model.init_hidden(1)
        optimizer.zero_grad()
        output, hidden = model(x, hidden)
        loss = criterion(output, y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, char_to_idx, idx_to_char, losses

def generate_text(model, char_to_idx, idx_to_char, start_char, length=50):
    model.eval()
    hidden = model.init_hidden(1)
    input_char = torch.tensor([[char_to_idx[start_char]]])
    generated = start_char
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_char, hidden)
            prob = nn.functional.softmax(output[-1], dim=0).data
            idx = torch.multinomial(prob, 1).item()
            generated += idx_to_char[idx]
            input_char = torch.tensor([[idx]])
    return generated

# ---------------------- 模块3：预训练架构对比（轻量模型） ----------------------
@st.cache_resource
def load_masked_lm():
    return pipeline("fill-mask", model="distilroberta-base")

@st.cache_resource
def load_causal_lm():
    return pipeline("text-generation", model="distilgpt2")

# ---------------------- 模块4：语言模型评价（困惑度PPL） ----------------------
@st.cache_resource
def load_ppl_model():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def calculate_ppl(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = torch.exp(loss).item()
    return ppl

# ---------------------- 页面内容 ----------------------
st.title("📚 语言模型训练与对比分析平台")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "模块1：n元语言模型与平滑",
    "模块2：RNN语言模型训练",
    "模块3：预训练架构对比",
    "模块4：困惑度PPL计算"
])

# ---------------------- 模块1：n元语言模型与平滑 ----------------------
with tab1:
    st.header("🔤 n元语言模型与数据平滑")
    st.markdown("基于内置语料训练Trigram模型，对比平滑前后的概率计算结果")
    
    n_gram = st.selectbox("选择n元语法", [2, 3], index=1, key="ngram_select")
    use_smoothing = st.checkbox("开启Laplace平滑", value=False, key="smoothing_check")
    input_sentence = st.text_area(
        "输入句子计算生成概率",
        value="The company reported a profit.",
        height=100,
        key="ngram_text"
    )
    
    if st.button("计算概率", key="ngram_btn"):
        with st.spinner("训练模型中..."):
            model = train_ngram_model(n=n_gram, use_smoothing=use_smoothing)
            prob = get_ngram_prob(model, input_sentence, n=n_gram)
            st.success(f"句子生成概率：{prob:.8f}")
            if prob == 0:
                st.warning("未开启平滑时，未见过的Trigram会导致概率为0（数据稀疏问题）")

# ---------------------- 模块2：从零训练RNN语言模型 ----------------------
with tab2:
    st.header("🔄 从零训练RNN语言模型")
    st.markdown("使用字符级RNN训练自定义文本，观察序列模式学习效果")
    
    train_text = st.text_area(
        "输入训练文本（建议短文本）",
        value="hello world hello python hello streamlit",
        height=150,
        key="rnn_text"
    )
    hidden_size = st.slider("隐藏层维度", 16, 128, 32, key="hidden_slider")
    epochs = st.slider("训练轮数", 10, 200, 50, key="epoch_slider")
    lr = st.slider("学习率", 0.001, 0.01, 0.005, step=0.001, key="lr_slider")
    
    if st.button("开始训练", key="rnn_btn"):
        with st.spinner("训练中..."):
            model, char_to_idx, idx_to_char, losses = train_char_rnn(train_text, hidden_size, epochs, lr)
            st.line_chart(pd.DataFrame(losses, columns=["Loss"]))
            st.session_state["rnn_model"] = (model, char_to_idx, idx_to_char)
            st.success("训练完成！")
    
    if "rnn_model" in st.session_state:
        start_char = st.text_input("输入起始字符", value="h", key="start_char")
        if st.button("生成文本", key="gen_btn"):
            model, char_to_idx, idx_to_char = st.session_state["rnn_model"]
            if start_char in char_to_idx:
                generated = generate_text(model, char_to_idx, idx_to_char, start_char)
                st.code(generated)
            else:
                st.error("起始字符不在训练文本中，请重新输入")

# ---------------------- 模块3：预训练架构对比 ----------------------
with tab3:
    st.header("⚖️ Masked LM vs. Causal LM")
    st.markdown("对比RoBERTa（双向）与DistilGPT2（单向）的生成机制差异")
    
    masked_lm = load_masked_lm()
    causal_lm = load_causal_lm()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RoBERTa（Masked LM）")
        masked_text = st.text_input(
            "输入带<mask>的句子",
            value="The man went to the <mask> to buy some milk.",
            key="masked_text"
        )
        if st.button("预测Mask", key="bert_btn"):
            with st.spinner("预测中..."):
                result = masked_lm(masked_text, top_k=5)
                st.dataframe(pd.DataFrame([{"token": r["token_str"], "score": r["score"]} for r in result]))
    
    with col2:
        st.subheader("DistilGPT2（Causal LM）")
        prompt_text = st.text_input(
            "输入提示词",
            value="The man went to the store to buy",
            key="prompt_text"
        )
        if st.button("续写文本", key="gpt_btn"):
            with st.spinner("生成中..."):
                result = causal_lm(prompt_text, max_new_tokens=20, do_sample=True)
                st.code(result[0]["generated_text"])

# ---------------------- 模块4：困惑度PPL计算 ----------------------
with tab4:
    st.header("📊 语言模型困惑度（PPL）计算")
    st.markdown("基于DistilGPT2计算句子困惑度，数值越小表示模型越匹配该句子")
    
    ppl_model, ppl_tokenizer = load_ppl_model()
    
    test_sentences = st.text_area(
        "输入测试句子（每行一句）",
        value="The quick brown fox jumps over the lazy dog.\nasd qwe zxc 123",
        height=150,
        key="ppl_text"
    )
    
    if st.button("计算PPL", key="ppl_btn"):
        sentences = [s.strip() for s in test_sentences.split("\n") if s.strip()]
        results = []
        for sent in sentences:
            ppl = calculate_ppl(ppl_model, ppl_tokenizer, sent)
            results.append({"句子": sent, "困惑度(PPL)": round(ppl, 2)})
        st.dataframe(pd.DataFrame(results))

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.markdown("© 2025 NLP 课程 Week X 实验 | 语言模型训练与对比分析平台")
