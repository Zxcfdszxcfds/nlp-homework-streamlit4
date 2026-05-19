
import streamlit as st
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="机器翻译对比与评测系统",
    page_icon="🌐",
    layout="wide"
)


# ---------------------- 缓存模型加载 ----------------------
@st.cache_resource(show_spinner="正在加载翻译模型...")
def load_translator():
    """加载 Hugging Face 的英译中模型"""
    translator = pipeline(
        "translation_en_to_zh",
        model="Helsinki-NLP/opus-mt-en-zh",
        device=-1  # 使用 CPU，避免无 GPU 报错
    )
    return translator


translator = load_translator()

# ---------------------- 基于规则的翻译词典 ----------------------
# 基础英中词典，模拟早期机器翻译
basic_dict = {
    "I": "我",
    "you": "你",
    "he": "他",
    "she": "她",
    "it": "它",
    "we": "我们",
    "they": "他们",
    "am": "是",
    "is": "是",
    "are": "是",
    "was": "是",
    "were": "是",
    "have": "有",
    "has": "有",
    "do": "做",
    "does": "做",
    "did": "做",
    "go": "去",
    "went": "去",
    "eat": "吃",
    "ate": "吃",
    "drink": "喝",
    "drank": "喝",
    "run": "跑",
    "ran": "跑",
    "walk": "走",
    "walked": "走",
    "like": "喜欢",
    "likes": "喜欢",
    "love": "爱",
    "loves": "爱",
    "cat": "猫",
    "dog": "狗",
    "rain": "下雨",
    "cats": "猫",
    "dogs": "狗",
    "raining": "下雨",
    "raining cats and dogs": "下猫下狗"  # 俚语的逐词保留
}


def rule_based_translate(sentence: str) -> str:
    """基于词典的逐词直译"""
    words = sentence.strip().split()
    translated = []
    for word in words:
        # 处理标点
        clean_word = word.strip(".,!?").lower()
        if clean_word in basic_dict:
            translated.append(basic_dict[clean_word])
        else:
            # 不在词典里的词直接保留
            translated.append(word)
    return " ".join(translated)


# ---------------------- 页面内容 ----------------------
st.title("🌐 机器翻译对比与评测系统")
st.markdown("---")

# 分三个模块的 Tab
tab1, tab2, tab3 = st.tabs([
    "模块1：神经机器翻译引擎",
    "模块2：直译 vs. 意译对比",
    "模块3：BLEU 自动评测"
])

# ---------------------- 模块1：神经机器翻译引擎 ----------------------
with tab1:
    st.header("🧠 神经机器翻译引擎 (NMT Engine)")
    st.markdown("输入英文句子，体验基于 Transformer 的英译中效果。")

    # 输入框
    en_text = st.text_area(
        "请输入英文句子：",
        value="It rains cats and dogs.",
        height=150
    )

    if st.button("开始翻译", key="btn1"):
        with st.spinner("模型正在翻译中..."):
            # 调用翻译模型
            result = translator(en_text)[0]["translation_text"]
            st.success("翻译完成！")
            st.subheader("译文结果：")
            st.info(result)

# ---------------------- 模块2：直译 vs. 意译对比 ----------------------
with tab2:
    st.header("⚖️ 基于规则的直译 vs. 神经网络意译")
    st.markdown("对比两种翻译范式的差异，观察基于规则翻译的局限性。")

    # 输入框
    en_text2 = st.text_area(
        "请输入英文句子：",
        value="It rains cats and dogs.",
        height=150
    )

    if st.button("开始对比", key="btn2"):
        with st.spinner("正在对比两种翻译结果..."):
            # 1. 基于规则的直译
            rule_trans = rule_based_translate(en_text2)
            # 2. 神经机器翻译
            nmt_trans = translator(en_text2)[0]["translation_text"]

            # 并排展示
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("基于规则的直译")
                st.warning(rule_trans)
            with col2:
                st.subheader("神经网络意译")
                st.success(nmt_trans)

# ---------------------- 模块3：BLEU 自动评测 ----------------------
with tab3:
    st.header("📊 机器翻译质量自动评测 (BLEU Score)")
    st.markdown("输入待评测译文和参考译文，自动计算 BLEU 分数（0~1，越高越接近参考译文）。")

    # 输入框
    candidate_text = st.text_area("待评测译文（如 NMT 或直译结果）：", height=100)
    reference_text = st.text_area("参考译文（人工翻译或标准译文）：", height=100)

    if st.button("计算 BLEU 分数", key="btn3"):
        if not candidate_text or not reference_text:
            st.error("请输入待评测译文和参考译文！")
        else:
            # 分词
            candidate = candidate_text.split()
            reference = [reference_text.split()]  # 参考译文需要是列表的列表

            # 计算 BLEU，带平滑函数避免零分
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

            st.success(f"BLEU 分数：{bleu_score:.4f}")
            # 解释分数
            if bleu_score >= 0.7:
                st.info("✅ 译文质量优秀，与参考译文高度匹配")
            elif bleu_score >= 0.4:
                st.info("⚠️ 译文质量中等，部分内容与参考译文有差异")
            else:
                st.warning("❌ 译文质量较差，与参考译文差异较大")

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.markdown("© 2025 NLP 课程 Week 9 实验 | 机器翻译对比与评测系统")