# BERTopic統合版テキスト分析アプリ
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# BERTopic関連のインポート
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    BERTOPIC_ERROR = str(e)

# ページ設定
st.set_page_config(
    page_title="BERTopic テキスト分析アプリ",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 BERTopic テキスト分析アプリ")
if BERTOPIC_AVAILABLE:
    st.markdown("**BERTopic対応版** - 高度なトピック分析が可能です！")
else:
    st.warning(f"⚠️ BERTopicが利用できません: {BERTOPIC_ERROR}")
    st.markdown("**軽量版モード** - 基本的な分析のみ利用可能")

# サイドバー設定
st.sidebar.header("⚙️ 分析パラメータ")

# 分析手法選択
if BERTOPIC_AVAILABLE:
    analysis_method = st.sidebar.selectbox(
        "分析手法",
        ["BERTopic（推奨）", "軽量版（K-means）"],
        help="BERTopicは高精度、軽量版は高速"
    )
else:
    analysis_method = "軽量版（K-means）"
    st.sidebar.info("BERTopicが利用できないため、軽量版を使用")

# 共通パラメータ
if analysis_method == "BERTopic（推奨）":
    n_topics = st.sidebar.selectbox(
        "トピック数",
        ["自動検出"] + list(range(5, 51, 5)),
        help="自動検出を推奨"
    )
    min_topic_size = st.sidebar.slider("最小トピックサイズ", 5, 50, 10)
    language = st.sidebar.selectbox("言語", ["英語", "日本語"])
else:
    n_clusters = st.sidebar.slider("クラスター数", 2, 20, 5)
    max_features = st.sidebar.slider("最大特徴数", 50, 500, 100)
    min_df = st.sidebar.slider("最小文書頻度", 1, 10, 2)

# メインエリア
tab1, tab2, tab3 = st.tabs(["📤 データ分析", "📊 結果可視化", "📈 高度な分析"])

with tab1:
    st.header("CSVファイルをアップロード")
    
    uploaded_file = st.file_uploader(
        "CSVファイルを選択してください",
        type=['csv'],
        help="分析したいテキストデータを含むCSVファイル"
    )
    
    if uploaded_file is not None:
        try:
            # ファイル読み込み
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            st.success(f"✅ ファイル読み込み完了: {len(df)}行")
            
            # データプレビュー
            st.subheader("データプレビュー")
            st.dataframe(df.head(), use_container_width=True)
            
            # 列選択
            col1, col2, col3 = st.columns(3)
            
            with col1:
                content_column = st.selectbox(
                    "分析対象テキスト列",
                    df.columns.tolist(),
                    help="分析したいテキストが含まれる列"
                )
            
            with col2:
                title_column = st.selectbox(
                    "タイトル列（オプション）",
                    ["なし"] + df.columns.tolist(),
                    help="文書のタイトルが含まれる列"
                )
            
            with col3:
                timestamp_column = st.selectbox(
                    "時系列列（オプション）",
                    ["なし"] + df.columns.tolist(),
                    help="年や日付の情報がある列"
                )
            
            # 分析実行ボタン
            if st.button("🚀 テキスト分析を実行", type="primary"):
                
                # プログレスバー
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 データ前処理中...")
                    progress_bar.progress(10)
                    
                    # テキストデータ取得
                    texts = df[content_column].dropna().astype(str).tolist()
                    
                    # BERTopic分析
                    if analysis_method == "BERTopic（推奨）" and BERTOPIC_AVAILABLE:
                        
                        min_cluster_size = max(min_topic_size, 2)
                        if len(texts) < min_cluster_size:
                            st.error(f"❌ データが不足しています。最低{min_cluster_size}件の文書が必要です。")
                            st.stop()
                        
                        status_text.text("🤖 BERTopicモデル初期化中...")
                        progress_bar.progress(30)
                        
                        # BERTopicモデル設定
                        if language == "英語":
                            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                            vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
                        else:
                            embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                            vectorizer = CountVectorizer(ngram_range=(1, 2))
                        
                        umap_model = UMAP(n_components=5, random_state=42)
                        
                        topic_model = BERTopic(
                            embedding_model=embedding_model,
                            umap_model=umap_model,
                            vectorizer_model=vectorizer,
                            nr_topics=None if n_topics == "自動検出" else n_topics,
                            min_topic_size=min_topic_size,
                            calculate_probabilities=True,
                            verbose=False
                        )
                        
                        status_text.text("📊 BERTopicトピック抽出中...")
                        progress_bar.progress(60)
                        
                        # トピック抽出
                        topics, probs = topic_model.fit_transform(texts)
                        
                        status_text.text("🎨 結果処理中...")
                        progress_bar.progress(80)
                        
                        # 結果の整理
                        topic_info = topic_model.get_topic_info()
                        
                        # 2D可視化用の座標計算
                        try:
                            umap_2d = UMAP(n_components=2, random_state=42)
                            embeddings = topic_model._extract_embeddings(texts)
                            coords_2d = umap_2d.fit_transform(embeddings)
                        except:
                            # フォールバック: TF-IDFを使用
                            vectorizer_fallback = TfidfVectorizer(max_features=100, stop_words='english')
                            tfidf_matrix = vectorizer_fallback.fit_transform(texts)
                            pca = PCA(n_components=2, random_state=42)
                            coords_2d = pca.fit_transform(tfidf_matrix.toarray())
                        
                        # セッション状態に結果を保存
                        st.session_state.bertopic_results = {
                            'method': 'BERTopic',
                            'topic_model': topic_model,
                            'topics': topics,
                            'probs': probs,
                            'topic_info': topic_info,
                            'coords_2d': coords_2d,
                            'texts': texts,
                            'df': df,
                            'content_column': content_column,
                            'title_column': title_column if title_column != "なし" else None,
                            'timestamp_column': timestamp_column if timestamp_column != "なし" else None
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("✅ BERTopic分析完了！")
                        
                        st.success(f"🎉 BERTopic分析完了！{len(topic_info)}個のトピックが検出されました")
                        
                        # トピック概要表示
                        st.subheader("📋 トピック概要")
                        st.dataframe(topic_info, use_container_width=True)
                    
                    else:
                        # 軽量版分析（K-means）
                        if len(texts) < n_clusters:
                            st.error(f"❌ データが不足しています。最低{n_clusters}件の文書が必要です。")
                            st.stop()
                        
                        status_text.text("🔤 テキストベクトル化中...")
                        progress_bar.progress(40)
                        
                        # TF-IDF ベクトル化
                        vectorizer = TfidfVectorizer(
                            max_features=max_features,
                            stop_words='english',
                            ngram_range=(1, 2),
                            min_df=min_df
                        )
                        
                        tfidf_matrix = vectorizer.fit_transform(texts)
                        feature_names = vectorizer.get_feature_names_out()
                        
                        status_text.text("📊 クラスタリング実行中...")
                        progress_bar.progress(70)
                        
                        # K-meansクラスタリング
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(tfidf_matrix)
                        
                        status_text.text("🎨 可視化データ生成中...")
                        progress_bar.progress(90)
                        
                        # PCAで2次元に次元削減（可視化用）
                        pca = PCA(n_components=2, random_state=42)
                        coords_2d = pca.fit_transform(tfidf_matrix.toarray())
                        
                        # セッション状態に結果を保存
                        st.session_state.bertopic_results = {
                            'method': 'K-means',
                            'clusters': clusters,
                            'coords_2d': coords_2d,
                            'tfidf_matrix': tfidf_matrix,
                            'feature_names': feature_names,
                            'kmeans': kmeans,
                            'texts': texts,
                            'df': df,
                            'n_clusters': n_clusters
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 軽量版分析完了！")
                        
                        st.success(f"🎉 軽量版分析完了！{n_clusters}個のクラスターが生成されました")
                    
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    import traceback
                    st.text("詳細なエラー情報:")
                    st.text(traceback.format_exc())
                    progress_bar.empty()
                    status_text.empty()
        
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {str(e)}")

with tab2:
    st.header("分析結果の可視化")
    
    if 'bertopic_results' in st.session_state:
        results = st.session_state.bertopic_results
        
        if results['method'] == 'BERTopic':
            # BERTopic結果の可視化
            st.subheader("📊 トピック分布")
            
            # トピックサイズのバーチャート
            topic_sizes = results['topic_info']['Count'].values
            topic_labels = [f"トピック {i}" for i in results['topic_info']['Topic'].values]
            
            fig_bar = px.bar(
                x=topic_labels,
                y=topic_sizes,
                title="各トピックの文書数",
                labels={'x': 'トピック', 'y': '文書数'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 2次元散布図
            st.subheader("🗺️ トピックマップ")
            
            scatter_df = pd.DataFrame({
                'x': results['coords_2d'][:, 0],
                'y': results['coords_2d'][:, 1],
                'topic': [f"トピック {t}" if t != -1 else "その他" for t in results['topics']],
                'text': [text[:100] + '...' if len(text) > 100 else text for text in results['texts']]
            })
            
            fig_scatter = px.scatter(
                scatter_df,
                x='x', y='y',
                color='topic',
                hover_data=['text'],
                title="文書のトピック分布マップ"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # トピック詳細
            st.subheader("🔤 トピック詳細")
            
            for _, row in results['topic_info'].iterrows():
                topic_id = row['Topic']
                if topic_id != -1:  # その他トピックを除く
                    with st.expander(f"トピック {topic_id} - {row['Name']} ({row['Count']}件)"):
                        
                        # キーワード表示
                        topic_words = results['topic_model'].get_topic(topic_id)
                        if topic_words:
                            keywords = [word for word, _ in topic_words[:10]]
                            st.write(f"**キーワード**: {', '.join(keywords)}")
                        
                        # 代表文書
                        topic_docs = [i for i, t in enumerate(results['topics']) if t == topic_id]
                        if topic_docs:
                            st.write("**代表文書**:")
                            for i, doc_idx in enumerate(topic_docs[:3]):
                                if doc_idx < len(results['texts']):
                                    st.write(f"{i+1}. {results['texts'][doc_idx][:200]}...")
            
        else:
            # K-means結果の可視化（従来通り）
            st.subheader("📊 クラスター分布")
            cluster_counts = pd.Series(results['clusters']).value_counts().sort_index()
            
            fig_bar = px.bar(
                x=[f"クラスター {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                title="各クラスターの文書数",
                labels={'x': 'クラスター', 'y': '文書数'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 2次元散布図
            st.subheader("🗺️ クラスター可視化")
            
            scatter_df = pd.DataFrame({
                'x': results['coords_2d'][:, 0],
                'y': results['coords_2d'][:, 1],
                'cluster': [f"クラスター {c}" for c in results['clusters']],
                'text': [text[:100] + '...' if len(text) > 100 else text for text in results['texts']]
            })
            
            fig_scatter = px.scatter(
                scatter_df,
                x='x', y='y',
                color='cluster',
                hover_data=['text'],
                title="文書のクラスター分布"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 結果ダウンロード
        st.subheader("💾 結果ダウンロード")
        
        # 結果データフレーム作成
        results_df = results['df'].copy()
        if results['method'] == 'BERTopic':
            results_df['topic'] = results['topics']
            results_df['topic_probability'] = [prob.max() if hasattr(prob, 'max') else prob for prob in results['probs']]
            results_df['topic_name'] = [f"トピック {t}" if t != -1 else "その他" for t in results['topics']]
        else:
            results_df['cluster'] = results['clusters']
            results_df['cluster_name'] = [f"クラスター {c}" for c in results['clusters']]
        
        # ダウンロードボタン
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 分析結果をCSVでダウンロード",
            data=csv,
            file_name=f"{results['method'].lower()}_analysis_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👆 まず「データ分析」タブでファイルをアップロードし、分析を実行してください。")

with tab3:
    st.header("高度な分析機能")
    
    if 'bertopic_results' in st.session_state and st.session_state.bertopic_results['method'] == 'BERTopic':
        results = st.session_state.bertopic_results
        
        # 時系列分析
        if results['timestamp_column']:
            st.subheader("📈 時系列トピック分析")
            
            try:
                # 時系列データの準備
                timestamps = results['df'][results['timestamp_column']].values
                topics_over_time = results['topic_model'].topics_over_time(
                    results['texts'], 
                    results['topics'], 
                    timestamps
                )
                
                # 時系列可視化
                fig_timeline = results['topic_model'].visualize_topics_over_time(topics_over_time)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
            except Exception as e:
                st.error(f"時系列分析でエラーが発生しました: {str(e)}")
        
        # トピック階層
        st.subheader("🌳 トピック階層分析")
        
        if st.button("階層分析を実行"):
            try:
                with st.spinner("階層分析を実行中..."):
                    hierarchical_topics = results['topic_model'].hierarchical_topics(results['texts'])
                    fig_hierarchy = results['topic_model'].visualize_hierarchy(hierarchical_topics=hierarchical_topics)
                    st.plotly_chart(fig_hierarchy, use_container_width=True)
            except Exception as e:
                st.error(f"階層分析でエラーが発生しました: {str(e)}")
        
        # トピック類似度
        st.subheader("🔗 トピック類似度マップ")
        
        if st.button("類似度分析を実行"):
            try:
                with st.spinner("類似度分析を実行中..."):
                    fig_heatmap = results['topic_model'].visualize_heatmap()
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"類似度分析でエラーが発生しました: {str(e)}")
    
    else:
        st.info("高度な分析機能はBERTopic分析結果でのみ利用可能です。")

# フッター
st.markdown("---")
if BERTOPIC_AVAILABLE:
    st.markdown("""
    **BERTopic テキスト分析アプリ** 
    - 高度なトピックモデリング対応
    - 時系列分析・階層分析機能
    - 多言語対応
    """)
else:
    st.markdown("""
    **軽量版テキスト分析アプリ** 
    - TF-IDF + K-means クラスタリング
    - BERTopic機能は準備中
    """)