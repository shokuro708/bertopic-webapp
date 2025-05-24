# Phase 1: 基本的なCSV分析機能のStreamlit版
import streamlit as st
import pandas as pd
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
import os
import tempfile
import zipfile
from io import BytesIO

# ページ設定
st.set_page_config(
    page_title="BERTopic Web分析ツール",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 BERTopic Web分析ツール")
st.markdown("---")

# サイドバーでパラメータ設定
st.sidebar.header("⚙️ 分析パラメータ")

# 言語設定
language = st.sidebar.selectbox(
    "言語モデル",
    ["japanese", "english"],
    help="分析対象テキストの言語を選択してください"
)

# N-gram設定
col1, col2 = st.sidebar.columns(2)
with col1:
    ngram_min = st.number_input("N-gram最小", value=1, min_value=1, max_value=3)
with col2:
    ngram_max = st.number_input("N-gram最大", value=2, min_value=1, max_value=5)

# トピック数
nr_topics = st.sidebar.selectbox(
    "トピック数",
    ["Auto"] + list(range(5, 101, 5)),
    help="生成するトピック数。Autoで自動決定"
)

# 最小トピックサイズ
min_topic_size = st.sidebar.slider(
    "最小トピックサイズ",
    min_value=5,
    max_value=100,
    value=10,
    help="トピックを形成する最小文書数"
)

# メインエリア
tab1, tab2, tab3 = st.tabs(["📤 ファイルアップロード", "🔧 詳細設定", "📊 結果表示"])

with tab1:
    st.header("CSVファイルをアップロード")
    
    uploaded_file = st.file_uploader(
        "CSVファイルを選択してください",
        type=['csv'],
        help="分析したいテキストデータを含むCSVファイル"
    )
    
    if uploaded_file is not None:
        # ファイル読み込み
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        st.success(f"✅ ファイル読み込み完了: {len(df)}行")
        
        # データプレビュー
        st.subheader("データプレビュー")
        st.dataframe(df.head(), use_container_width=True)
        
        # 列選択
        st.subheader("分析対象列の選択")
        col1, col2 = st.columns(2)
        
        with col1:
            content_column = st.selectbox(
                "コンテンツ列（分析対象テキスト）",
                df.columns.tolist(),
                help="分析したいテキストが含まれる列"
            )
        
        with col2:
            title_column = st.selectbox(
                "タイトル列",
                ["なし"] + df.columns.tolist(),
                help="文書のタイトルが含まれる列（オプション）"
            )

with tab2:
    st.header("詳細設定")
    
    # 詳細パラメータを展開式で表示
    with st.expander("🎯 トピック抽出設定"):
        diversity = st.slider("多様性", 0.0, 1.0, 0.5, 0.1)
        seed_words = st.text_input("シード単語（カンマ区切り）", "")
        seed_multiplier = st.number_input("シード乗数", value=2.0, min_value=1.0, max_value=10.0)
    
    with st.expander("🧹 前処理設定"):
        custom_stop_words = st.text_area(
            "カスタムストップワード（カンマ区切り）",
            value="method,device,apparatus,system,process,unit,component,element"
        )
    
    with st.expander("📈 出力設定"):
        top_n_words = st.slider("トピック当たりキーワード数", 5, 20, 10)
        representative_docs = st.slider("代表文書数", 5, 20, 10)

with tab3:
    st.header("分析結果")
    
    if 'df' in locals() and content_column:
        
        # 分析実行ボタン
        if st.button("🚀 BERTopic分析を実行", type="primary"):
            
            # プログレスバー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 分析処理（簡略化版）
                status_text.text("🔄 データ前処理中...")
                progress_bar.progress(20)
                
                # テキストデータ取得
                texts = df[content_column].dropna().astype(str).tolist()
                
                if len(texts) < min_topic_size:
                    st.error(f"❌ データが不足しています。最低{min_topic_size}件の文書が必要です。")
                    st.stop()
                
                status_text.text("🤖 BERTopicモデル作成中...")
                progress_bar.progress(40)
                
                # BERTopicモデル作成（簡略化）
                topic_model = BERTopic(
                    language="japanese" if language == "japanese" else "english",
                    nr_topics=None if nr_topics == "Auto" else int(nr_topics),
                    min_topic_size=min_topic_size,
                    calculate_probabilities=True
                )
                
                status_text.text("📊 トピック抽出中...")
                progress_bar.progress(70)
                
                # トピック抽出
                topics, probs = topic_model.fit_transform(texts)
                
                status_text.text("📈 結果生成中...")
                progress_bar.progress(90)
                
                # 結果表示
                progress_bar.progress(100)
                status_text.text("✅ 分析完了！")
                
                # トピック情報表示
                st.success(f"🎉 分析完了！{len(set(topics))}個のトピックが検出されました")
                
                # トピック概要
                topic_info = topic_model.get_topic_info()
                st.subheader("📋 トピック概要")
                st.dataframe(topic_info, use_container_width=True)
                
                # トピック可視化
                st.subheader("🗺️ トピックマップ")
                fig1 = topic_model.visualize_topics()
                st.plotly_chart(fig1, use_container_width=True)
                
                # トピック分布
                st.subheader("📊 トピック分布")
                fig2 = topic_model.visualize_barchart(top_k_topics=10)
                st.plotly_chart(fig2, use_container_width=True)
                
                # 結果ダウンロード
                st.subheader("💾 結果ダウンロード")
                
                # 結果CSVを作成
                results_df = df.copy()
                results_df['topic'] = topics
                results_df['topic_probability'] = [prob.max() if len(prob.shape) > 1 else prob for prob in probs]
                
                # トピックキーワードを追加
                topic_keywords = {}
                for topic_id in set(topics):
                    if topic_id != -1:
                        keywords = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
                        topic_keywords[topic_id] = ", ".join(keywords)
                    else:
                        topic_keywords[topic_id] = "その他"
                
                results_df['topic_keywords'] = results_df['topic'].map(topic_keywords)
                
                # ダウンロードボタン
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 分析結果をCSVでダウンロード",
                    data=csv_buffer.getvalue(),
                    file_name="bertopic_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# フッター
st.markdown("---")
st.markdown("**BERTopic Web分析ツール** - Powered by Streamlit")
