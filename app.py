import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import folium
import json
import warnings
import streamlit as st
import os
from visuals import picture, navigation, route_map, uri
from datetime import datetime

warnings.filterwarnings('ignore')

# ── 1. Initialize Client ──
from pathlib import Path
from html import escape
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Beyond Encore · Hong Kong", page_icon=str(Path(__file__).resolve().parent / "assets" / "beyond-encore-logo.png"), layout="wide")
BASE = Path(__file__).resolve().parent
DATA = Path(os.getenv("POI_DATA_DIR", str(BASE / "data")))
def setting(name, default=""):
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default

api_key = setting("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url=setting("OPENAI_BASE_URL", "https://api.openai.com/v1"), timeout=30, max_retries=1) if api_key else None

# ── 2. Load Data ──
# Path definitions
tag_path = str(DATA / "tags")
info_path = str(DATA / "filter")


poi_dfs = {}
poi_types = ["restaurant", "attraction", "bar", "cafe", "night_club", "shopping_mall"]

@st.cache_data
def load_all_poi_data():
    local_poi_dfs = {}
    for poi_type in poi_types:
        try:
            import sqlite3
            database = Path(os.getenv("POI_DATABASE_PATH", str(DATA / "poi.sqlite3")))
            with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True) as connection:
                combined_df = pd.read_sql_query(f'SELECT * FROM "{poi_type}"', connection)

            local_poi_dfs[poi_type] = combined_df
        
        except Exception as e:
            st.error(f"Failed to load {poi_type} data: {e}")
            local_poi_dfs[poi_type] = pd.DataFrame()
    
    return local_poi_dfs

# ── 3. Configuration Mapping ──
venue_coords = {
    "Hong Kong Coliseum": (22.3015, 114.1820),
    "West Kowloon Cultural District": (22.3010, 114.1550),
    "Hong Kong Cultural Centre": (22.2941, 114.1704),
    "Kai Tak Sports Park": (22.3224, 114.1972)
}

PRICE_DESC_TO_LEVEL = {
    "Free ($0)": 0, "Affordable ($0-$80)": 1, "Middle ($80-$250)": 2, "High ($250-$800)": 3, "Luxury (>$800)": 4
}
LEVEL_TO_PRICE_DESC = {v: k for k, v in PRICE_DESC_TO_LEVEL.items()}

# ── 4. Utility Functions ──
def haversine_vec(lat1, lng1, lat2, lng2):
    R = 6371
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * R

def get_embeddings_batch(text_list, batch_size=20):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

def poi_to_text(row):
    return f"{row.get('poi_name','')} | Suitable for music: {row.get('music_style','')} | Fan scenarios: {row.get('fan_scenario','')}"

# ── Compute POI embeddings on startup (only once) ──
@st.cache_resource
def precompute_embeddings(_all_dfs):
    # Deploy only precomputed vectors; never trigger bulk paid API calls at startup.
    cache_file = DATA / "poi_embeddings.npz"
    if not cache_file.exists():
        return {}
    with np.load(cache_file, allow_pickle=False) as cache:
        return {category: {"matrix": cache[category + "_matrix"],
                           "place_ids": cache[category + "_ids"].tolist()}
                for category in poi_types if category + "_matrix" in cache}

# Execute global initialization
poi_dfs = load_all_poi_data()
poi_embeddings_dict = precompute_embeddings(poi_dfs)

# ── Core Functions ──
def parse_concert_input(user_input):
    prompt = f"""
    User entered a concert name: "{user_input}"
    Based on your knowledge of this band/artist, return the following JSON format (return only JSON, no other content):
    {{
        "artist": "Artist/band name",
        "music_type": "Music genre, e.g., pop/rock/electronic/hip-hop",
        "tags": "Style tags, comma-separated, e.g., energetic, nostalgic, emotional"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def get_scenario_template(user_purpose, group_size, time_pref):
    """Generate tag chain based on user input"""
    dining_tag = "Solo Dining" if group_size == "Alone" else "Group Dining"

    # Define scenario template chains
    templates = {
        "Concert Warm-up": [
            {"type": "shopping_mall", "tag": "Pre-concert Prep", "desc": "Store bags here, buy water supplies"},
            {"type": "attraction", "tag": "Fan Support", "desc": "Feel the fan support atmosphere outside the venue, exchange fan cards"},
            {"type": "restaurant", "tag": "Group Dining", "desc": "Meal time, join fan gatherings here"},
            {"type": "bar", "tag": "Support Activities", "desc": "Fan-organized warm-up activities, chat and drink for better atmosphere"}
        ],
        "Celebrity Trail": [
            {"type": "attraction", "tag": "Pilgrimage", "desc": "Celebrity filming locations, strike the same pose"},
            {"type": "shopping_mall", "tag": "Celebrity Hunt", "desc": "Check out stores visited by celebrities for similar items"},
            {"type": "cafe", "tag": "Photo Spot", "desc": "Have coffee where the celebrity sat"},
            {"type": "restaurant", "tag": "Support Activities", "desc": "Meal time, join fan gatherings here"}
        ],
        "Sightseeing Experience": [
            {"type": "attraction", "tag": "Photogenic Spot", "desc": "Hong Kong landmarks, must-visit list priority"},
            {"type": "cafe", "tag": "Photo Spot", "desc": "Experience unique cafes, feel the local lifestyle"},
            {"type": "attraction", "tag": "Sunset/Night View", "desc": "Watch the sunset at scenic spots in the evening"},
            {"type": "restaurant", "tag": "Photo Spot", "desc": "Eat at unique restaurants, take photos"}
        ],
        "Extreme Commando": [
            {"type": "attraction", "tag": "Transport Hub", "desc": "Near transportation hubs for easy access in/out of city"},
            {"type": "shopping_mall", "tag": "All-in-one Supply", "desc": "One-stop solution for shopping and fast food"},
            {"type": "attraction", "tag": "Photogenic Spot", "desc": "Must-see landmark check-in"},
            {"type": "restaurant", "tag": "Solo Dining", "desc": "Eat quickly and continue journey, efficiency first"}
        ],
        "Healing Solo Trip": [
            {"type": "cafe", "tag": "Solo Dining", "desc": "Quiet environment, perfect for relaxing alone"},
            {"type": "attraction", "tag": "Pilgrimage", "desc": "Celebrity filming locations, experience idol's life moments"},
            {"type": "restaurant", "tag": "Solo Dining", "desc": "Portion and atmosphere suitable for solo travelers"},
            {"type": "bar", "tag": "Photo Spot", "desc": "Unique bars, perfect for solo drinks"}
        ],
        "Culinary Journey": [
            {"type": "restaurant", "tag": "Group Dining", "desc": "Meal time, join fan gatherings here"},
            {"type": "attraction", "tag": "Photogenic Spot", "desc": "Hong Kong landmarks, must-visit list priority"},
            {"type": "cafe", "tag": "Photo Spot", "desc": "Experience unique cafes, feel the local lifestyle"},
            {"type": "restaurant", "tag": "Late Night Eats", "desc": "Enjoy midnight snacks, experience Hong Kong nightlife culture"}
        ]
    }
    result = [dict(step) for step in templates.get(user_purpose, templates["Concert Warm-up"])]
    for step in result:
        if step["type"] == "restaurant" and step["tag"] in ("Group Dining", "Solo Dining"):
            step["tag"] = dining_tag
    return result
    
# Route combination recommendation
def generate_itinerary_global(all_poi_dfs, template, venue_coord, price_range, target_hour, target_weekday, top_n):
    """
    top_n: Value from Streamlit slider, determines how many POIs the final route includes
    """
    itinerary = []
    current_pos = venue_coord
    
    # ── Dynamically adjust template length ──
    # If top_n is longer than template, cycle through template; if shorter, truncate
    if top_n <= len(template):
        active_template = template[:top_n]
    else:
        # E.g., template has 4 items, user wants 6, cycle to complete: 1,2,3,4,1,2
        active_template = (template * ((top_n // len(template)) + 1))[:top_n]

    # ── Dynamic weight calculation ──
    # When there are more POIs, moderately reduce distance weight (from 0.8 to 0.6), increase rating weight
    dist_weight = 0.5 if top_n <= 3 else 0.2
    rating_weight = 1.0 - dist_weight

    for i, step in enumerate(active_template):
        df = all_poi_dfs.get(step["type"], pd.DataFrame()).copy()
        if df.empty: continue
        
        df = df.dropna(subset=['lat', 'lng']).copy()
        if target_hour is not None and 'day_of_week' in df.columns:
            df = df[df['day_of_week'] == target_weekday]
            overnight = df['is_overnight'].astype(str).str.lower().isin(['true', '1'])
            regular = (df['open_decimal'] <= target_hour) & (target_hour < df['close_decimal'])
            late = overnight & (df['close_decimal'] < df['open_decimal']) & ((target_hour >= df['open_decimal']) | (target_hour < df['close_decimal']))
            df = df[regular | late]
        df = df.drop_duplicates(subset=['place_id'])
        # Exclude place_ids already in itinerary to prevent repetition
        used_ids = [item['place_id'] for item in itinerary]
        df = df[~df['place_id'].isin(used_ids)]

        # --- Hierarchical filtering logic ---
        mask_tag = df['fan_scenario'].str.contains(step['tag'], na=False)
        mask_price = (df['price_level'] >= price_range[0]) & (df['price_level'] <= price_range[1]) if 'price_level' in df.columns else True
        candidates = df[mask_tag & mask_price].copy()
        if candidates.empty: candidates = df[mask_price].copy()
        if candidates.empty: continue

        # --- Calculate scores ---
        candidates['dist_to_prev'] = haversine_vec(current_pos[0], current_pos[1], candidates['lat'].values, candidates['lng'].values)
        
        # Use dynamic weights
        candidates['route_score'] = (1 / (candidates['dist_to_prev'] + 0.5)) * dist_weight + (candidates['rating'] / 5) * rating_weight
        
        if candidates.empty: continue
        
        best_row = candidates.nlargest(1, 'route_score').iloc[0]
        res_item = best_row.to_dict()
        res_item['step_desc'] = step['desc']
        res_item['category'] = step['type']
        res_item['step_tag'] = step['tag'] # Dynamic tagging
        
        itinerary.append(res_item)
        current_pos = (best_row['lat'], best_row['lng'])
        
    return itinerary

# ── 5. Core Recommendation Logic ──
def recommend_from_user_input(user_input, venue="Hong Kong Coliseum", top_n=5, price_range=(1,4), 
                               target_hour=None, target_weekday=None,
                               weight_sim=0.5, weight_dist=0.25, weight_rating=0.2, weight_review=0.05):
    
    if client is None:
        raise ValueError("Set OPENAI_API_KEY in the deployment environment to enable concert matching.")
    parsed = parse_concert_input(user_input)
    
    # Get query vector
    text = f"""
    A live music concert by {parsed['artist']},
    music genre: {parsed['music_type']},
    featuring styles and themes such as {parsed['tags']}.
    """
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    query_embedding = np.array(embedding_response.data[0].embedding)

    coords = venue_coords.get(venue, (None, None))
    if coords[0] is None: return None, parsed, {}

    m = folium.Map(location=[coords[0], coords[1]], zoom_start=14)
    folium.Marker(location=coords, popup=f"🎤 {user_input}", icon=folium.Icon(color="red", icon="music", prefix="fa")).add_to(m)

    all_results = {}
    poi_colors = {
        "restaurant":    "blue",
        "attraction":    "green",
        "bar":           "purple",
        "cafe":          "orange",
        "night_club":    "darkred",
        "shopping_mall": "gray",
    }

    for poi_type, df in poi_dfs.items():
        if df.empty: continue
        
        # A. Basic filtering
        if 'price_level' in df.columns:
            df = df[(df['price_level'] >= price_range[0]) & (df['price_level'] <= price_range[1])]
        
        # B. Business hours filtering
        if target_hour is not None and 'day_of_week' in df.columns:
            df = df[df['day_of_week'] == target_weekday]
            def is_open(row):
                if str(row.get('is_overnight', False)).lower() in ('true', '1') and row['close_decimal'] < row['open_decimal']:
                    return target_hour >= row['open_decimal'] or target_hour <= row['close_decimal']
                return row['open_decimal'] <= target_hour <= row['close_decimal']
            df = df[df.apply(is_open, axis=1)]

        # C. Deduplication
        df = df.drop_duplicates(subset=['place_id']).copy()
        if df.empty: continue

        # D. Similarity calculation (using precise index matching)
        stored_data = poi_embeddings_dict.get(poi_type)
        if not stored_data: continue
        
        full_ids = stored_data["place_ids"]
        id_to_idx = {pid: i for i, pid in enumerate(full_ids)}
        indices = [id_to_idx[pid] for pid in df["place_id"].tolist() if pid in id_to_idx]
        
        if not indices: continue
        
        df = df[df["place_id"].isin(id_to_idx)].copy()
        relevant_matrix = stored_data["matrix"][indices]
        df["sim_score"] = cosine_similarity([query_embedding], relevant_matrix)[0]

        # E. Comprehensive scoring
        df["distance"] = haversine_vec(coords[0], coords[1], df["lat"].values, df["lng"].values)
        max_dist = df["distance"].max() if df["distance"].max() > 0 else 1
        df["dist_score"] = 1 - df["distance"] / max_dist
        df["rating_score"] = df["rating"].fillna(0) / 5.0
        df["review_score"] = np.log1p(df["user_rating_count"].fillna(0))
        df["review_score"] /= (df["review_score"].max() if df["review_score"].max() > 0 else 1)
        
        df["final_score"] = (weight_sim * df["sim_score"] + weight_dist * df["dist_score"] + 
                             weight_rating * df["rating_score"] + weight_review * df["review_score"])
        top = df.nlargest(top_n, "final_score")
        if not top.empty:
            all_results[poi_type] = top 
            
            # (Optional) Add points to map
            for _, row in top.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lng"]],
                    radius=5,
                    color=poi_colors.get(poi_type, "blue"),
                    fill=True,
                    popup=f"{row['name']} (Score: {row['final_score']:.2f})"
                ).add_to(m)

    return m, parsed, all_results

# Beyond Encore interface
language = st.sidebar.selectbox("语言 / Language", ["中文", "English"], key="language")
ZH = language == "中文"
TRANSLATIONS = {'Make it your night': '今晚，由你定义', 'Choose your mood, company and budget.': '选择旅行心情、同行伙伴和预算。', 'Your trip mood': '旅行主题', 'Concert Warm-up': '演唱会预热', 'Celebrity Trail': '追星打卡', 'Sightseeing Experience': '城市漫游', 'Extreme Commando': '高效特种兵之旅', 'Healing Solo Trip': '独自疗愈', 'Culinary Journey': '美食探索', "Who's coming?": '和谁一起出发？', 'Alone': '一个人', 'With Others': '与朋友同行', 'Opening hours': '营业时间', 'Unlimited': '不限时间', 'Current Time': '现在', 'Custom': '自定义', 'Visit date': '出行日期', 'Visit time': '出行时间', 'All opening-hour filters use Hong Kong time.': '营业时间按香港当地时间筛选。', 'Budget per stop · HKD': '每站预算 · 港币', 'Free ($0)': '免费（$0）', 'Affordable ($0-$80)': '实惠（$0–$80）', 'Middle ($80-$250)': '适中（$80–$250）', 'High ($250-$800)': '高档（$250–$800）', 'Luxury (>$800)': '奢享（>$800）', '01 / Start with your concert': '01 / 从你的演唱会开始', 'Artist or concert': '艺人或演唱会名称', 'e.g. Mayday Hong Kong Concert': '例如：五月天香港演唱会', 'Concert venue': '演出场馆', 'Stops to explore': '想探索几站？', 'Find my encore →': '开启我的安可之旅 →', 'AI matching is ready to configure. You can preview an itinerary now; add your API key in the deployment environment to enable music matching.': '当前可预览按距离与评分推荐的行程；配置 API 密钥后，即可启用音乐风格匹配。', 'Enter an artist or concert name to begin.': '请先输入艺人或演唱会名称。', 'Finding your next favourite places…': '正在寻找你的下一处心动地点…', "We couldn't complete music matching. Check the API configuration and connection, then try again.": '暂时无法完成音乐匹配，请检查 API 配置和网络后重试。', 'Your evening, in three steps': '三步，开启精彩的一晚', '01 · Set the scene': '01 · 选定演出', '02 · Find your places': '02 · 找到心仪地点', '03 · Keep exploring': '03 · 继续探索', 'Choose your concert and venue.': '输入演唱会并选择场馆。', 'Tune your trip mood, budget and opening hours.': '按心情、预算和营业时间定制行程。', 'Explore your itinerary on the map and save your stops.': '在地图上查看路线，保存你的行程。', '02 / Your encore itinerary': '02 / 你的安可行程', 'Artist': '艺人', 'Places to explore': '探索地点', 'Straight-line distance': '直线总距离', 'No places match these filters. Try a wider budget or select Unlimited opening hours.': '没有符合条件的地点，请扩大预算范围或选择不限时间。', '03 / Explore the neighbourhood': '03 / 探索场馆周边', 'Save itinerary · CSV': '保存行程 · CSV', 'Unknown': '未知', 'Open in Maps ↗': '在地图中打开 ↗', 'Hong Kong Coliseum': '香港体育馆（红馆）', 'West Kowloon Cultural District': '西九文化区', 'Hong Kong Cultural Centre': '香港文化中心', 'Kai Tak Sports Park': '启德体育园', 'Pre-concert Prep': '演前准备', 'Fan Support': '粉丝应援', 'Group Dining': '朋友聚餐', 'Support Activities': '应援活动主题', 'Pilgrimage': '主题打卡', 'Celebrity Hunt': '追星探索', 'Photo Spot': '拍照打卡', 'Photogenic Spot': '城市出片地', 'Sunset/Night View': '日落与夜景', 'Transport Hub': '交通枢纽', 'All-in-one Supply': '一站补给', 'Solo Dining': '独自用餐', 'Late Night Eats': '深夜美食', 'BEYOND ENCORE / HONG KONG': 'BEYOND ENCORE / 香港', 'The show ends.': '演出落幕，', "Your night doesn't.": '精彩还在继续', 'Turn your concert into a Hong Kong adventure. Discover food, hidden corners and places to unwind — all around your venue.': '让一场演唱会，成为一趟香港奇遇。在场馆周边发现美食、城市角落与放松好去处。', '6 kinds of places': '六类探索地点', '4 concert venues': '四大演出场馆', 'Your own itinerary': '专属你的行程', 'Connections show straight-line distances, not walking directions. Fan-themed tags do not verify celebrity visits or organised events.': '连线为直线距离，并非步行导航。粉丝主题标签不代表已核实的明星到访或应援活动。', 'BEYOND ENCORE · A little more Hong Kong, after the music. Ratings, prices and opening hours reflect the supplied dataset.': 'BEYOND ENCORE · 音乐之后，再多感受一点香港。评分、价格及营业时间来自现有数据集。', 'restaurant': '餐厅', 'attraction': '景点', 'bar': '酒吧', 'cafe': '咖啡馆', 'night club': '夜店', 'shopping mall': '购物中心', 'Name': '名称', 'Rating': '评分', 'Distance (km)': '距离（公里）', 'Address': '地址'}
def tr(text):
    return TRANSLATIONS.get(text, text) if ZH else text
def tr_html(text):
    css, separator, content = text.partition('</style>')
    if not separator:
        content, css = text, ''
    if ZH:
        for en, zh in sorted(TRANSLATIONS.items(), key=lambda item: -len(item[0])):
            content = content.replace(en, zh)
    return css + separator + content

st.markdown(f'<div class="brand-header"><img src="{uri("beyond-encore-logo.png")}" alt="Beyond Encore logo"><div><strong>Beyond Encore</strong><span>{"音乐之后，探索继续" if ZH else "More to explore, after the music"}</span></div></div>', unsafe_allow_html=True)
st.markdown(tr_html("""<style>
.stApp {background:#0e1220;color:#f4f2ed}
.block-container {max-width:1320px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#171c2c;border-right:1px solid #30364a}
h1,h2,h3 {letter-spacing:-.035em}
.hero {padding:44px 40px;border:1px solid #41445e;border-radius:24px;background:radial-gradient(ellipse at 95% 0%,#474270,transparent 60%),linear-gradient(120deg,#20263d,#14192a);margin-bottom:26px}
.eyebrow {color:#c6bcff;font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
.hero h1 {font-size:clamp(38px,6vw,68px);line-height:1.05;margin:18px 0}
.hero p {color:#c2c6d6;font-size:17px;max-width:640px;line-height:1.7}
.pill {display:inline-block;border:1px solid #53566e;border-radius:99px;padding:6px 12px;margin:6px 6px 0 0;color:#e1dcff;font-size:12px}
.stop {border:1px solid #363c52;border-radius:18px;padding:22px;background:#1a2031;min-height:240px;margin-bottom:16px}
.stop h3 {font-size:22px;margin:14px 0}.stop p {color:#b9c1d5;font-size:14px;line-height:1.6}
.stop a {color:#d6caff}.stop .number {color:#beafff;font-size:12px;font-weight:700;letter-spacing:.1em}
[data-testid="stMetric"] {background:#1a2031;border:1px solid #363c52;border-radius:14px;padding:16px}
[data-testid="stMetricValue"] {font-size:22px;white-space:normal}
.stButton>button {border-radius:12px;min-height:48px;font-weight:700}
@media(max-width:640px){.hero{padding:28px 22px}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
<div class="hero"><div class="eyebrow">BEYOND ENCORE / HONG KONG</div>
<h1>The show ends.<br>Your night doesn't.</h1>
<p>Turn your concert into a Hong Kong adventure. Discover food, hidden corners and places to unwind — all around your venue.</p>
<span class="pill">6 kinds of places</span><span class="pill">4 concert venues</span><span class="pill">Your own itinerary</span></div>
"""), unsafe_allow_html=True)

st.markdown("""<style>
.stApp{background:#faf7f1;color:#30343b}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#f2e7e6,#e9eff2);border-right:1px solid #e2dcda}
.hero{background:linear-gradient(120deg,#f5e5e4,#e9eff2 70%,#edf0e7);border:1px solid #e9dcdb;box-shadow:0 12px 40px #826e6a0a}
.hero h1{color:#343638}.hero p{color:#717276}.eyebrow{color:#b2868a}.pill{background:#fff9;border-color:#ddd3d0;color:#66666b}
.stop{background:#fff;border:1px solid #eee7e3;box-shadow:0 5px 16px #5b4b4010;color:#35383c;overflow:hidden}.stop p{color:#73777b}
.stop a{color:#7b96a4}.stop .number{color:#b58d91}
[data-testid="stMetric"]{background:#fff;border:1px solid #e8e3df;box-shadow:0 4px 15px #60483908}
.poi-card{padding:0;min-height:0}.card-body{padding:22px}.poi-photo{height:200px;position:relative;background-color:#e8eeee}.poi-photo span{position:absolute;right:10px;bottom:10px;font-size:10px;background:#fffD;color:#666;padding:4px 9px;border-radius:30px}
.concert-photo{height:220px;border-radius:18px;margin-bottom:20px}.stButton>button{border-radius:99px}
.timeline{display:flex;gap:16px;margin:12px 0;align-items:center}.timeline-number{flex:none;width:36px;height:36px;border-radius:50%;background:#cda3a5;color:white;display:grid;place-items:center}.timeline-body{background:#fff;border:1px solid #eee7e3;border-radius:15px;padding:14px 18px;flex:1}.timeline-body small{display:block;color:#899397;margin-top:6px}
@media(max-width:640px){.poi-photo{height:180px}.hero h1{font-size:38px}}

.stApp{background:radial-gradient(ellipse at 5% 15%,#f0d9d77d,transparent 45%),radial-gradient(ellipse at 92% 50%,#dce9efb0,transparent 50%),#faf7f1;background-attachment:fixed}
.brand-header{display:flex;align-items:center;gap:14px;margin:0 0 22px;padding:6px 0}
.brand-header img{width:68px;height:61px;object-fit:contain;border-radius:16px}
.brand-header strong{display:block;font-size:23px;letter-spacing:-.03em;color:#41484b}
.brand-header span{display:block;font-size:12px;letter-spacing:.04em;color:#859298;margin-top:4px}
.hero{background:linear-gradient(120deg,#fff7,#f1e5e459,#e0ebef66);backdrop-filter:blur(22px) saturate(125%);-webkit-backdrop-filter:blur(22px) saturate(125%);border:1px solid #ffffffcf;box-shadow:inset 0 1px 0 #fff,0 12px 36px #7a6c6510}
.hero h1{font-size:clamp(30px,4vw,48px);line-height:1!important;margin:16px 0 20px;letter-spacing:-.025em}
.stop,.timeline-body,[data-testid="stMetric"]{background:linear-gradient(135deg,#ffffffb5,#ffffff5e);backdrop-filter:blur(20px) saturate(130%);-webkit-backdrop-filter:blur(20px) saturate(130%);border:1px solid #ffffffdf;box-shadow:inset 0 1px 0 #ffffffed,0 8px 28px #687e8812}
.card-body{background:linear-gradient(125deg,#ffffff60,#e9eff235)}
.pill{background:#fff6;border-color:#ffffffb0;backdrop-filter:blur(10px)}
.stop{transition:transform .2s ease,box-shadow .2s ease}
.stop:hover{transform:translateY(-3px);box-shadow:inset 0 1px 0 #fff,0 14px 34px #687e8822}
@media(prefers-reduced-motion:reduce){.stop{transition:none}.stop:hover{transform:none}}
@media(max-width:640px){.hero h1{font-size:30px;line-height:1!important}.brand-header img{width:56px;height:50px}.brand-header strong{font-size:21px}}
</style>""", unsafe_allow_html=True)

st.sidebar.title(tr("Make it your night"))
st.sidebar.caption(tr("Choose your mood, company and budget."))
user_purpose = st.sidebar.selectbox(tr("Your trip mood"), ["Concert Warm-up", "Celebrity Trail", "Sightseeing Experience", "Extreme Commando", "Healing Solo Trip", "Culinary Journey"], format_func=tr, key="trip_mood")
group_size = st.sidebar.radio(tr("Who's coming?"), ["Alone", "With Others"], horizontal=True, format_func=tr, key="company")
time_pref = "Regular Hours"
time_option = st.sidebar.radio(tr("Opening hours"), ["Unlimited", "Current Time", "Custom"], format_func=tr, key="hours")
now = datetime.now(ZoneInfo("Asia/Hong_Kong"))
target_hour, target_weekday = None, now.weekday()
if time_option == "Current Time":
    target_hour = now.hour + now.minute / 60
elif time_option == "Custom":
    d = st.sidebar.date_input(tr("Visit date"), value=now.date())
    t = st.sidebar.time_input(tr("Visit time"), value=now.time().replace(tzinfo=None))
    target_hour, target_weekday = t.hour + t.minute / 60, d.weekday()
st.sidebar.caption(tr("All opening-hour filters use Hong Kong time."))
selected_prices = st.sidebar.select_slider(tr("Budget per stop · HKD"), options=list(PRICE_DESC_TO_LEVEL), value=("Affordable ($0-$80)", "Luxury (>$800)"), format_func=tr, key="budget")
price_range = tuple(PRICE_DESC_TO_LEVEL[x] for x in selected_prices)
st.subheader(tr("01 / Start with your concert"))
with st.form("concert_form"):
    c1, c2 = st.columns([3, 2])
    user_input = c1.text_input(tr("Artist or concert"), placeholder=tr("e.g. Mayday Hong Kong Concert"), key="concert")
    venue = c2.selectbox(tr("Concert venue"), list(venue_coords), format_func=tr, key="venue")
    top_n = st.slider(tr("Stops to explore"), 1, 10, 5, key="stops")
    submitted = st.form_submit_button(tr("Find my encore →"), type="primary", use_container_width=True)
if not api_key:
    st.info(tr("AI matching is ready to configure. You can preview an itinerary now; add your API key in the deployment environment to enable music matching."))
if submitted:
    if not user_input.strip():
        st.warning(tr("Enter an artist or concert name to begin."))
    else:
        try:
            with st.spinner(tr("Finding your next favourite places…")):
                if client:
                    _, parsed, res = recommend_from_user_input(user_input, venue, top_n, price_range, target_hour, target_weekday)
                else:
                    parsed, res = {"artist": user_input, "music_type": "Preview · distance & rating"}, {}
                itinerary = generate_itinerary_global(poi_dfs, get_scenario_template(user_purpose, group_size, time_pref), venue_coords[venue], price_range, target_hour, target_weekday, top_n)
                st.session_state['plan'] = dict(parsed=parsed, res=res, itinerary=itinerary, venue=venue, purpose=user_purpose)
        except Exception:
            st.error(tr("We couldn't complete music matching. Check the API configuration and connection, then try again."))
if 'plan' not in st.session_state:
    st.subheader(tr("Your evening, in three steps"))
    for col, title, body in zip(st.columns(3), [tr("01 · Set the scene"), tr("02 · Find your places"), tr("03 · Keep exploring")], [tr("Choose your concert and venue."), tr("Tune your trip mood, budget and opening hours."), tr("Explore your itinerary on the map and save your stops.")]):
        with col:
            st.markdown(f'<div class="stop"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.subheader("探索你的下一站" if ZH else "Discover your next stop")
    category_names={'restaurant':('餐厅','Food & friends'),'attraction':('景点','City & harbour'),'bar':('酒吧','One more toast'),'cafe':('咖啡馆','Coffee moments'),'night_club':('夜生活','After the show'),'shopping_mall':('购物中心','A little retail therapy')}
    for row in [poi_types[:3], poi_types[3:]]:
        for col, category in zip(st.columns(3), row):
            with col:
                title=category_names[category][0 if ZH else 1]
                st.markdown(f'<div class="stop poi-card">{picture(category, ZH)}<div class="card-body"><h3>{title}</h3><p>{poi_dfs[category]["place_id"].nunique()} {"个地点" if ZH else "places"}</p></div></div>',unsafe_allow_html=True)
else:
    plan = st.session_state['plan']
    itinerary = plan['itinerary']
    st.subheader(tr("02 / Your encore itinerary"))
    c1, c2, c3 = st.columns(3)
    c1.metric(tr("Artist"), plan['parsed']['artist'])
    c2.metric(tr("Places to explore"), len(itinerary))
    c3.metric(tr("Straight-line distance"), f"{sum(s.get('dist_to_prev', 0) for s in itinerary):.1f} km")
    st.caption(f"{tr(plan['venue'])} · {tr(plan['purpose'])} · {tr("Connections show straight-line distances, not walking directions. Fan-themed tags do not verify celebrity visits or organised events.")}")
    if not itinerary:
        st.info(tr("No places match these filters. Try a wider budget or select Unlimited opening hours."))
    else:
        for offset in range(0, len(itinerary), 2):
            cols = st.columns(2)
            for j, spot in enumerate(itinerary[offset:offset+2]):
                val = spot.get('price_level')
                price = LEVEL_TO_PRICE_DESC.get(int(float(val)), tr('Unknown')) if pd.notna(val) else tr('Unknown')
                name = escape(str(spot.get('name', 'Place')))
                address = escape(str(spot.get('formatted_address', '')))
                with cols[j]:
                    st.markdown(f'<div class="stop poi-card">{picture(spot.get("category", "attraction"), ZH)}<div class="card-body"><div class="number">{"第" if ZH else "STOP "}{offset+j+1:02d}{"站" if ZH else ""} / {escape(tr(str(spot["step_tag"])))}</div><h3>{name}</h3><p>{address}</p><p>★ {spot.get("rating", "—")} · {escape(tr(price))}</p><a href="{escape(navigation(spot), quote=True)}" target="_blank" rel="noopener noreferrer">{tr("Open in Maps ↗")}</a></div></div>', unsafe_allow_html=True)
        st.subheader(tr("03 / Explore the neighbourhood"))
        route_map(venue_coords[plan['venue']], tr(plan['venue']), itinerary, ZH)
        st.caption("地图：香港地政总署 · 连线为直线示意，不代表实际步行路线。" if ZH else "Map: Lands Department, Hong Kong · Connections are straight lines, not walking routes.")
        export = pd.DataFrame(itinerary)
        fields = [x for x in ['name', 'formatted_address', 'rating', 'price_level', 'lat', 'lng', 'step_tag'] if x in export]
        st.download_button(tr("Save itinerary · CSV"), export[fields].to_csv(index=False).encode('utf-8-sig'), 'beyond-encore-itinerary.csv', 'text/csv')
    if itinerary:
        st.subheader("行程时间线" if ZH else "Your route timeline")
        for i, spot in enumerate(itinerary):
            distance=float(spot.get('dist_to_prev',0))
            meta=f"距上一站 {distance:.2f} 公里 · 直线距离" if ZH else f"{distance:.2f} km from previous stop · straight line"
            st.markdown(f'<div class="timeline"><div class="timeline-number">{i+1}</div><div class="timeline-body"><strong>{escape(str(spot["name"]))}</strong><small>{meta}</small></div></div>',unsafe_allow_html=True)
    for category, df in plan['res'].items():
        with st.expander(f"{'更多推荐' if ZH else 'More matches'} · {tr(category.replace('_', ' '))} · {len(df)}"):
            fields = [x for x in ['name', 'rating', 'distance', 'formatted_address'] if x in df]
            st.dataframe(df[fields].rename(columns={"name": tr("Name"), "rating": tr("Rating"), "distance": tr("Distance (km)"), "formatted_address": tr("Address")}), use_container_width=True, hide_index=True)
st.divider()
st.caption(tr("BEYOND ENCORE · A little more Hong Kong, after the music. Ratings, prices and opening hours reflect the supplied dataset."))
