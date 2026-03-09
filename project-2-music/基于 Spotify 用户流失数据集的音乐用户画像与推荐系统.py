# ===================== 模块1：环境配置与数据加载（无额外依赖） =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
import random

# 全局设置（解决中文显示、忽略警告）
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 加载数据集（确保df一定被定义）
file_path = 'spotify_churn_dataset.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"✅ 数据集加载完成：{df.shape[0]:,}行，{df.shape[1]}列")
else:
    print(f"⚠️ 未找到{file_path}，自动生成模拟数据（可正常运行）")
    n_rows = 10000
    df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_rows)],
        'song_id': [f'song_{i}' for i in range(n_rows)],
        'song_name': [f'歌曲{i}' for i in range(n_rows)],
        'artist': [f'歌手{random.randint(1, 100)}' for _ in range(n_rows)],
        'play_count': np.random.randint(1, 20, n_rows),
        'genre': np.random.choice(['Pop', 'Rock', 'Hip-Hop', 'Classical'], n_rows),
        'danceability': np.random.uniform(0, 1, n_rows),
        'energy': np.random.uniform(0, 1, n_rows),
        'valence': np.random.uniform(0, 1, n_rows),
        'tempo': np.random.uniform(60, 180, n_rows)
    })

# ===================== 模块2：数据清洗（保留核心字段，适配推荐场景） =====================
print("\n" + "=" * 50)
print("开始数据清洗...")

# 1. 定义核心字段（确保数据集包含这些信息，无则自动补充）
core_cols = ['user_id', 'song_id', 'song_name', 'artist', 'play_count', 'genre',
             'danceability', 'energy', 'valence', 'tempo']
for col in core_cols:
    if col not in df.columns:
        if col == 'play_count':
            df[col] = 1  # 播放次数默认1
        elif col in ['danceability', 'energy', 'valence']:
            df[col] = np.random.uniform(0, 1, len(df))  # 音频特征默认0-1随机值
        elif col == 'tempo':
            df[col] = np.random.uniform(60, 180, len(df))  # 节奏默认60-180随机值
        else:
            df[col] = 'Unknown'  # 文本字段默认值

# 2. 强制转换数值型列，从源头避免类型错误
df['play_count'] = pd.to_numeric(df['play_count'], errors='coerce').fillna(0)
df['danceability'] = pd.to_numeric(df['danceability'], errors='coerce').fillna(0.5)
df['energy'] = pd.to_numeric(df['energy'], errors='coerce').fillna(0.5)
df['valence'] = pd.to_numeric(df['valence'], errors='coerce').fillna(0.5)
df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce').fillna(120)

# 3. 清洗规则：删缺失值、去重、筛有效数据
df_clean = df[core_cols].copy()
df_clean = df_clean.dropna(subset=['user_id', 'song_id'])  # 删核心字段缺失值
df_clean = df_clean.drop_duplicates(subset=['user_id', 'song_id'])  # 删用户-歌曲重复记录
df_clean = df_clean[df_clean['play_count'] >= 1]  # 仅保留播放过的记录
# 音频特征值限定在合理范围
df_clean = df_clean[df_clean['danceability'].between(0, 1)]
df_clean = df_clean[df_clean['energy'].between(0, 1)]
df_clean = df_clean[df_clean['valence'].between(0, 1)]
df_clean = df_clean[df_clean['tempo'].between(60, 180)]

df_clean = df_clean.reset_index(drop=True)
print(f"✅ 数据清洗完成：{df_clean.shape[0]:,}行有效数据")

# ===================== 模块3：用户画像构建（K-Means聚类，无额外依赖） =====================
print("\n" + "=" * 50)
print("开始用户画像聚类...")

# 1. 分两步计算用户特征，彻底避免类型问题
# 第一步：计算数值型特征（播放行为 + 音频特征）
user_num_features = df_clean.groupby('user_id').agg(
    total_play_count=('play_count', 'sum'),
    avg_play_count=('play_count', 'mean'),
    played_song_count=('play_count', 'count'),
    avg_danceability=('danceability', 'mean'),
    avg_energy=('energy', 'mean'),
    avg_valence=('valence', 'mean'),
    avg_tempo=('tempo', 'mean')
).reset_index()


# 第二步：单独计算流派众数（字符串类型，不参与数值聚合）
def get_mode(series):
    if series.empty:
        return 'Pop'
    mode_values = series.mode()
    return mode_values.iloc[0] if not mode_values.empty else 'Pop'


user_genre = df_clean.groupby('user_id')['genre'].apply(get_mode).reset_index(name='prefer_genre')

# 第三步：合并两部分特征，得到最终用户画像数据
user_features = pd.merge(user_num_features, user_genre, on='user_id', how='left')

# 2. 特征标准化（聚类必备，消除量纲影响）
cluster_features = [
    'total_play_count', 'avg_play_count', 'played_song_count',
    'avg_danceability', 'avg_energy', 'avg_valence'
]
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features[cluster_features])

# 3. 手肘法选最佳聚类数（可视化）
sse = []
k_range = range(2, 8)  # 测试2-7个聚类
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init=10避免警告
    kmeans.fit(user_features_scaled)
    sse.append(kmeans.inertia_)

# 绘制手肘图（保存本地）
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'o-', color='#2ca02c', linewidth=2)
plt.title('手肘法确定用户画像聚类数（Spotify数据集）', fontsize=14, fontweight='bold')
plt.xlabel('聚类数k', fontsize=12)
plt.ylabel('SSE（误差平方和）', fontsize=12)
plt.xticks(k_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('用户画像聚类手肘图.png', dpi=300)
plt.show()
print("✅ 手肘图已保存：用户画像聚类手肘图.png")

# 4. 最佳聚类（k=5，分5类用户）
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
user_features['cluster'] = kmeans.fit_predict(user_features_scaled)


# 5. 定义用户画像标签（业务化解读）
def get_user_portrait(row):
    high_play_thresh = user_features['total_play_count'].quantile(0.7)  # 高播放量阈值（前30%）
    if row['total_play_count'] >= high_play_thresh:
        if row['avg_energy'] >= 0.7 and row['avg_valence'] >= 0.7:
            return '高频动感音乐爱好者'
        elif row['avg_energy'] <= 0.4 and row['avg_valence'] <= 0.4:
            return '高频舒缓音乐爱好者'
        else:
            return '高频多元音乐爱好者'
    else:
        if row['prefer_genre'] in ['Pop', 'Rock', 'Hip-Hop']:
            return '低频流行摇滚爱好者'
        else:
            return '低频小众流派爱好者'


user_features['user_portrait'] = user_features.apply(get_user_portrait, axis=1)

# 6. 画像分布统计与可视化
portrait_count = user_features['user_portrait'].value_counts()
portrait_rate = (portrait_count / portrait_count.sum() * 100).round(2)
portrait_result = pd.DataFrame({
    '用户画像': portrait_count.index,
    '用户数': portrait_count.values,
    '占比(%)': portrait_rate.values
})
print("\n📊 用户画像分群结果：")
print(portrait_result)

# 绘制画像分布饼图
plt.figure(figsize=(9, 9))
plt.pie(
    portrait_result['占比(%)'],
    labels=portrait_result['用户画像'],
    autopct='%1.1f%%',
    colors=sns.color_palette('Set2'),
    textprops={'fontsize': 10}
)
plt.title('Spotify用户画像分布占比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('用户画像分布饼图.png', dpi=300)
plt.show()
print("✅ 画像分布饼图已保存：用户画像分布饼图.png")

# ===================== 模块4：推荐系统（纯Pandas实现，无额外依赖） =====================
print("\n" + "=" * 50)
print("构建基于用户相似度的推荐系统...")


# 1. 播放次数映射为评分（1-5分，模拟用户偏好）
def map_play_to_rating(play_count):
    if play_count >= 10:
        return 5
    elif play_count >= 5:
        return 4
    elif play_count >= 3:
        return 3
    elif play_count >= 2:
        return 2
    else:
        return 1


df_clean['rating'] = df_clean['play_count'].apply(map_play_to_rating)

# 2. 构建用户-歌曲评分矩阵（推荐系统核心输入）
# 先过滤掉只有1首歌的情况，确保矩阵有意义
unique_songs = df_clean['song_id'].nunique()
if unique_songs < 2:
    print("⚠️ 数据集中歌曲数量不足，无法构建有效评分矩阵，将直接推荐热门歌曲")
    # 直接生成热门歌曲列表，跳过相似度计算
    popular_songs = df_clean.groupby('song_id')['rating'].mean().sort_values(ascending=False)
    rating_matrix = None  # 标记为None，后续判断
else:
    rating_matrix = df_clean.pivot_table(
        index='user_id',
        columns='song_id',
        values='rating',
        fill_value=0  # 未播放的歌曲评分为0
    )
    # 筛选有效用户（至少评过分的用户）
    rating_matrix = rating_matrix[rating_matrix.sum(axis=1) > 0]
    print(f"✅ 评分矩阵构建完成：{rating_matrix.shape[0]}个用户 × {rating_matrix.shape[1]}首歌曲")


# 3. 计算用户相似度（皮尔逊相关系数，纯Pandas实现）
def calculate_user_similarity(target_user_id, rating_matrix_local):
    """计算目标用户与其他用户的相似度"""
    if rating_matrix_local is None or target_user_id not in rating_matrix_local.index:
        return pd.Series(dtype='float64')  # 若目标用户不在矩阵中，返回空

    # 目标用户的评分向量
    target_user_ratings = rating_matrix_local.loc[target_user_id]

    # 计算与其他用户的皮尔逊相似度（排除自身）
    user_similarity = rating_matrix_local.corrwith(target_user_ratings, axis=1)
    user_similarity = user_similarity.drop(target_user_id, errors='ignore')  # 删自身
    user_similarity = user_similarity.dropna()  # 删无相似度的用户

    return user_similarity


# 4. 个性化推荐函数（基于相似用户偏好）
def recommend_songs(target_user_id, top_n=10):
    """
    给目标用户推荐Top-N歌曲
    target_user_id：目标用户ID
    top_n：推荐歌曲数量
    """
    # 步骤1：获取目标用户已听过的歌曲
    user_played_songs = df_clean[df_clean['user_id'] == target_user_id]['song_id'].unique()

    # 初始化变量，避免赋值前引用
    similar_user_songs = pd.DataFrame()
    recommend_song_ids = []

    # 步骤2：计算用户相似度，取前20个最相似用户
    user_similarity = calculate_user_similarity(target_user_id, rating_matrix)
    if user_similarity.empty:
        # 若无相似用户，推荐热门歌曲
        popular_songs = df_clean.groupby('song_id')['rating'].mean().sort_values(ascending=False)
        recommend_song_ids = popular_songs.drop(user_played_songs, errors='ignore').head(top_n).index.tolist()
    else:
        # 取前20个最相似用户
        similar_users = user_similarity.nlargest(20).index
        # 相似用户喜欢的歌曲（排除目标用户已听过的）
        similar_user_songs = df_clean[
            (df_clean['user_id'].isin(similar_users)) &
            (~df_clean['song_id'].isin(user_played_songs))
            ]
        # 按相似用户评分均值排序
        recommend_song_ids = similar_user_songs.groupby('song_id')['rating'].mean(
        ).sort_values(ascending=False).head(top_n).index.tolist()

    # 处理推荐歌曲ID为空的情况（兜底：推荐任意未听过的歌曲）
    if not recommend_song_ids:
        all_songs = df_clean['song_id'].unique()
        recommend_song_ids = [s for s in all_songs if s not in user_played_songs][:top_n]

    # 步骤3：匹配歌曲详情（确保包含song_id列）
    recommend_details = df_clean[df_clean['song_id'].isin(recommend_song_ids)][
        ['song_id', 'song_name', 'artist', 'genre', 'danceability', 'energy', 'rating']
    ].drop_duplicates(subset=['song_id']).head(top_n)

    # 补充推荐评分（基于相似用户均值）
    recommend_ratings = []
    for song_id in recommend_details['song_id'].values:
        if not similar_user_songs.empty and song_id in similar_user_songs['song_id'].values:
            avg_rating = similar_user_songs[similar_user_songs['song_id'] == song_id]['rating'].mean()
        else:
            avg_rating = df_clean[df_clean['song_id'] == song_id]['rating'].mean()
        recommend_ratings.append(round(avg_rating, 2))

    recommend_details['推荐评分'] = recommend_ratings
    return recommend_details[['song_name', 'artist', 'genre', 'danceability', 'energy', '推荐评分']]


# 5. 测试推荐功能（取数据集中第一个用户）
sample_user_id = user_features['user_id'].iloc[0]
recommend_result = recommend_songs(sample_user_id, top_n=10)
print(f"\n🎵 给用户ID={sample_user_id}推荐的Top10歌曲：")
print(recommend_result)

# 保存推荐结果到Excel
recommend_file = f'用户{sample_user_id}_个性化推荐结果.xlsx'
recommend_result.to_excel(recommend_file, index=False)
print(f"\n✅ 推荐结果已保存：{recommend_file}")
# ===================== 模块5：核心业务结论（适配腾讯音乐场景） =====================
print("\n" + "=" * 60)
print("🎯 项目核心业务结论（Spotify→腾讯音乐迁移价值）")
print("=" * 60)
top_portrait = portrait_result.iloc[0]['用户画像']
top_portrait_rate = portrait_result.iloc[0]['占比(%)']
print(f"1. 用户结构：{top_portrait}占比最高（{top_portrait_rate}%），是腾讯音乐核心运营人群，可重点推送该类音乐内容")
print(f"2. 推荐逻辑：基于用户相似度的协同过滤，无需复杂库即可落地，适合中小规模用户推荐场景")
print(f"3. 业务落地：可结合用户画像推出『动感专区』『舒缓歌单』，用推荐系统提升用户听歌时长与留存率")
print("=" * 60)
print("\n🎉 项目二运行完成！生成文件清单：")
print("1. 用户画像聚类手肘图.png")
print("2. 用户画像分布饼图.png")
print(f"3. {recommend_file}")