# 確実に動く軽量版テキスト分析アプリ
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ページ設定
st.set_page_config(
    page_title="BERTopic風テキスト分析アプリ",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 BERTopic風テキスト分析アプリ")
st.markdown("**PyTorch依存なしの軽量版** - 確実に動作します！")

# サイドバー設定
st.sidebar.header("⚙️ 分析パラメータ")
n_clusters = st.sidebar.slider("クラスター数", 2, 20, 5)
max_features = st.sidebar.slider("最大特徴数", 50, 500, 100)
min_df = st.sidebar.slider("最小文書頻度", 1, 10, 2)

# メインエリア
tab1, tab2 = st.tabs(["📤 データ分析", "📊 結果可視化"])

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
            col1, col2 = st.columns(2)
            
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
            
            # 分析実行ボタン
            if st.button("🚀 テキスト分析を実行", type="primary"):
                
                # プログレスバー
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 データ前処理中...")
                    progress_bar.progress(20)
                    
                    # テキストデータ取得
                    texts = df[content_column].dropna().astype(str).tolist()
                    
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
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 分析完了！")
                    
                    # セッション状態に結果を保存
                    st.session_state.analysis_results = {
                        'clusters': clusters,
                        'coords_2d': coords_2d,
                        'tfidf_matrix': tfidf_matrix,
                        'feature_names': feature_names,
                        'kmeans': kmeans,
                        'texts': texts,
                        'df': df
                    }
                    
                    st.success(f"🎉 分析完了！{n_clusters}個のクラスターが生成されました")
                    
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {str(e)}")

with tab2:
    st.header("分析結果の可視化")
    
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # クラスター分布
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
        st.subheader("🗺️ クラスター可視化（2次元マップ）")
        
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
            title="文書のクラスター分布（PCA次元削減）"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # クラスター別キーワード
        st.subheader("🔤 各クラスターの特徴キーワード")
        
        for cluster_id in range(len(set(results['clusters']))):
            with st.expander(f"クラスター {cluster_id} の詳細"):
                cluster_docs = results['tfidf_matrix'][results['clusters'] == cluster_id]
                if cluster_docs.shape[0] > 0:
                    # クラスター中心の計算
                    cluster_center = cluster_docs.mean(axis=0).A1
                    top_indices = cluster_center.argsort()[-15:][::-1]
                    top_words = [results['feature_names'][i] for i in top_indices]
                    
                    st.write(f"**キーワード**: {', '.join(top_words)}")
                    st.write(f"**文書数**: {cluster_docs.shape[0]}件")
                    
                    # サンプル文書
                    sample_indices = np.where(results['clusters'] == cluster_id)[0][:3]
                    st.write("**サンプル文書**:")
                    for i, idx in enumerate(sample_indices):
                        if idx < len(results['texts']):
                            st.write(f"{i+1}. {results['texts'][idx][:200]}...")
        
        # 結果ダウンロード
        st.subheader("💾 結果ダウンロード")
        
        # 結果データフレーム作成
        results_df = results['df'].copy()
        results_df['cluster'] = results['clusters']
        results_df['cluster_name'] = [f"クラスター {c}" for c in results['clusters']]
        
        # ダウンロードボタン
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 分析結果をCSVでダウンロード",
            data=csv,
            file_name="text_analysis_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👆 まず「データ分析」タブでファイルをアップロードし、分析を実行してください。")

# フッター
st.markdown("---")
st.markdown("""
**軽量版テキスト分析アプリ** 
- TF-IDF + K-means クラスタリング使用
- PyTorch不要で高速動作
- BERTopic版は準備中
""")