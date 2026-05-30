# app.py — Ultimate K-Means Customer Segmentation UI
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="K-Means Clustering | Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PREMIUM CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f0c29 100%);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Neon glow text */
    .neon-text {
        color: #00f5d4;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.5), 0 0 40px rgba(0, 245, 212, 0.3);
    }
    
    .neon-text-purple {
        color: #c77dff;
        text-shadow: 0 0 20px rgba(199, 125, 255, 0.5);
    }
    
    /* Animated gradient border */
    .gradient-border {
        position: relative;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.03);
        overflow: hidden;
    }
    
    .gradient-border::before {
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(45deg, #00f5d4, #c77dff, #00f5d4);
        border-radius: 26px;
        z-index: -1;
        animation: rotate 4s linear infinite;
        opacity: 0.6;
    }
    
    @keyframes rotate {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background: rgba(10, 10, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    div[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00f5d4 0%, #00c9a7 100%) !important;
        color: #0a0a1a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 32px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 245, 212, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(0, 245, 212, 0.5) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00f5d4, #c77dff) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 10px;
    }
    
    .stRadio > div > label {
        color: #ccd6f6 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 245, 212, 0.1) !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 245, 212, 0.3);
        box-shadow: 0 10px 40px rgba(0, 245, 212, 0.1);
    }
    
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 13px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    div[data-testid="stMetric"] div {
        color: #00f5d4 !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 20px !important;
        overflow: hidden !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #c77dff 0%, #9d4edd 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 14px 30px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(199, 125, 255, 0.3) !important;
    }
    
    /* Headers */
    h1 { font-size: 3em !important; font-weight: 800 !important; letter-spacing: -1px !important; }
    h2 { font-size: 2em !important; font-weight: 700 !important; }
    h3 { font-size: 1.5em !important; font-weight: 600 !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #00f5d4; border-radius: 4px; }
    
    /* Pulse animation for loading */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 245, 212, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 245, 212, 0.6); }
    }
    
    .pulse {
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    /* Floating particles effect */
    .particles {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv('Mall_Customers.csv')

try:
    df = load_data()
    data_loaded = True
except:
    st.error("⚠️ Please ensure `Mall_Customers.csv` is in the same folder!")
    st.stop()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# SIDEBAR — GLASSMORPHISM NAV
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <div style="font-size: 4em; margin-bottom: 10px;">🎯</div>
        <h2 style="color: #00f5d4; margin: 0; font-size: 1.8em;">K-Means</h2>
        <p style="color: #8892b0; font-size: 13px; margin-top: 5px; letter-spacing: 2px; text-transform: uppercase;">Customer Segmentation</p>
        <div style="width: 50px; height: 3px; background: linear-gradient(90deg, #00f5d4, #c77dff); margin: 15px auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation with icons
    page = st.radio("", [
        "🏠 Home",
        "📊 Data Overview",
        "🔍 Elbow Method",
        "🎯 Clustering",
        "📋 Results"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Settings panel
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius: 20px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
        <h4 style="color: #c77dff; margin-bottom: 15px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">⚙️ Configuration</h4>
    """, unsafe_allow_html=True)
    
    optimal_k = st.slider("Clusters (K)", 2, 10, 5, 
                         help="Number of customer segments")
    
    st.markdown("""
        <p style="color: #8892b0; font-size: 11px; margin-top: 10px;">
            💡 Tip: Use Elbow Method to find optimal K
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <p style="color: #8892b0; font-size: 11px; letter-spacing: 1px;">Task 02</p>
        <p style="color: #00f5d4; font-size: 12px; font-weight: 600;">Infotech Internship</p>
        <div style="width: 30px; height: 2px; background: #c77dff; margin: 10px auto; border-radius: 1px;"></div>
        <p style="color: #555; font-size: 10px;">Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# DARK PLOT HELPER
# ============================================
def create_dark_figure(figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_facecolor('#0f0f23')
    
    # Spine colors
    for spine in ax.spines.values():
        spine.set_color('#1e1e3f')
        spine.set_linewidth(1.5)
    
    # Ticks
    ax.tick_params(colors='#8892b0', labelsize=11)
    ax.xaxis.label.set_color('#ccd6f6')
    ax.yaxis.label.set_color('#ccd6f6')
    ax.title.set_color('#00f5d4')
    
    # Grid
    ax.grid(True, alpha=0.15, color='#4a5568', linestyle='--', linewidth=0.5)
    
    return fig, ax

# ============================================
# HOME PAGE — HERO SECTION
# ============================================
if page == "🏠 Home":
    # Hero
    st.markdown("""
    <div style="text-align: center; padding: 60px 0 50px 0;">
        <div style="display: inline-block; position: relative;">
            <h1 style="font-size: 4em; margin-bottom: 15px; background: linear-gradient(135deg, #00f5d4, #c77dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                Customer Segmentation
            </h1>
            <div style="position: absolute; bottom: -10px; left: 50%; transform: translateX(-50%); width: 100px; height: 4px; background: linear-gradient(90deg, #00f5d4, #c77dff); border-radius: 2px;"></div>
        </div>
        <p style="color: #8892b0; font-size: 1.3em; margin-top: 25px; max-width: 600px; margin-left: auto; margin-right: auto;">
            AI-powered K-Means clustering to discover hidden customer segments in retail data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    st.markdown('<div style="margin-bottom: 40px;">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    
    metrics_data = [
        ("👥 Total Customers", f"{len(df):,}", "#00f5d4"),
        ("📊 Features", f"{len(df.columns)}", "#c77dff"),
        ("💰 Avg Income", f"${df['Annual Income (k$)'].mean():.0f}k", "#ff6b6b"),
        ("⭐ Avg Score", f"{df['Spending Score (1-100)'].mean():.0f}", "#ffd93d")
    ]
    
    for col, (label, value, color) in zip([m1, m2, m3, m4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 3px solid {color};">
                <p style="color: #8892b0; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">{label}</p>
                <h2 style="color: {color}; font-size: 2.5em; margin: 0; font-weight: 800;">{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("### ✨ Explore the App")
    
    features = [
        ("📊", "Data Overview", "Interactive exploration with distributions and statistics", "#00f5d4"),
        ("🔍", "Elbow Method", "Find optimal K with automated WCSS analysis", "#c77dff"),
        ("🎯", "Clustering", "Visualize customer segments with centroids", "#ff6b6b"),
        ("📋", "Results", "Export segmented data for business decisions", "#ffd93d")
    ]
    
    cols = st.columns(2)
    for i, (icon, title, desc, color) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 20px; border-left: 4px solid {color}; transition: all 0.3s;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 2em; margin-right: 15px;">{icon}</span>
                    <h4 style="color: {color}; margin: 0; font-size: 1.3em;">{title}</h4>
                </div>
                <p style="color: #8892b0; font-size: 14px; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # CTA
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 40px;">
        <p style="color: #8892b0; margin-bottom: 20px; font-size: 1.1em;">Ready to discover customer segments?</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Analysis", use_container_width=False):
        st.session_state.page = "📊 Data Overview"
        st.rerun()

# ============================================
# DATA OVERVIEW
# ============================================
elif page == "📊 Data Overview":
    st.markdown("""
    <div style="padding: 20px 0 30px 0;">
        <h1 style="display: inline; background: linear-gradient(135deg, #00f5d4, #c77dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📊 Data Overview</h1>
        <p style="color: #8892b0; margin-top: 10px;">Explore the Mall Customers dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #00f5d4; margin-bottom: 15px;'>📋 Customer Records</h3>", unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(cmap='viridis', subset=['Annual Income (k$)', 'Spending Score (1-100)']),
            use_container_width=True,
            height=450
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Stats panel
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h3 style="color: #c77dff; margin-bottom: 20px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px;">📈 Key Statistics</h3>
        """, unsafe_allow_html=True)
        
        stats = [
            ("Average Age", f"{df['Age'].mean():.1f} years", "#00f5d4"),
            ("Age Range", f"{df['Age'].min()}-{df['Age'].max()}", "#4ecca3"),
            ("Female Customers", f"{(df['Gender']=='Female').sum()}", "#ff6b6b"),
            ("Male Customers", f"{(df['Gender']=='Male').sum()}", "#4d96ff"),
            ("Income Range", f"${df['Annual Income (k$)'].min()}k-${df['Annual Income (k$)'].max()}k", "#ffd93d"),
            ("Score Range", f"{df['Spending Score (1-100)'].min()}-{df['Spending Score (1-100)'].max()}", "#c77dff")
        ]
        
        for label, value, color in stats:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #8892b0; font-size: 13px;">{label}</span>
                <span style="color: {color}; font-weight: 700; font-size: 14px;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Gender distribution mini chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ff6b6b; margin-bottom: 15px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;'>⚥ Gender Split</h3>", unsafe_allow_html=True)
        
        fig, ax = create_dark_figure(figsize=(6, 4))
        gender_counts = df['Gender'].value_counts()
        colors_pie = ['#ff6b6b', '#4d96ff']
        wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie,
                                          startangle=90, textprops={'color': 'white', 'fontsize': 12})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
        ax.set_title('Gender Distribution', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution charts
    st.markdown("### 📊 Feature Distributions")
    
    c1, c2, c3 = st.columns(3)
    
    distributions = [
        (c1, 'Age', '#00f5d4', 'Age Distribution'),
        (c2, 'Annual Income (k$)', '#c77dff', 'Income Distribution'),
        (c3, 'Spending Score (1-100)', '#ff6b6b', 'Spending Score')
    ]
    
    for col, feature, color, title in distributions:
        with col:
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            fig, ax = create_dark_figure(figsize=(5, 4))
            
            # Gradient fill histogram
            n, bins, patches = ax.hist(df[feature], bins=20, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Add mean line
            mean_val = df[feature].mean()
            ax.axvline(mean_val, color='white', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.1f}')
            
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(facecolor='#0f0f23', edgecolor=color, labelcolor='white')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ELBOW METHOD
# ============================================
elif page == "🔍 Elbow Method":
    st.markdown("""
    <div style="padding: 20px 0 30px 0;">
        <h1 style="display: inline; background: linear-gradient(135deg, #c77dff, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔍 Elbow Method</h1>
        <p style="color: #8892b0; margin-top: 10px;">Finding the optimal number of clusters</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Computing WCSS for K=1 to 10..."):
        wcss = []
        K_range = range(1, 11)
        
        for k_val in K_range:
            km = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
    
    # Main chart
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    fig, ax = create_dark_figure(figsize=(14, 8))
    
    # Glow effect line
    ax.plot(K_range, wcss, 'o-', color='#00f5d4', linewidth=4, markersize=12, 
            markerfacecolor='#0a0a1a', markeredgecolor='#00f5d4', markeredgewidth=3, zorder=5)
    
    # Fill area under curve
    ax.fill_between(K_range, wcss, alpha=0.1, color='#00f5d4')
    
    # Optimal K line
    ax.axvline(x=optimal_k, color='#ff6b6b', linestyle='--', linewidth=3, alpha=0.9)
    ax.text(optimal_k, max(wcss)*0.95, f'  Selected K={optimal_k}', color='#ff6b6b', 
            fontsize=14, fontweight='bold', va='top')
    
    # Elbow annotation
    elbow_k = 5  # Typical elbow point
    ax.annotate('Elbow Point', xy=(elbow_k, wcss[elbow_k-1]), 
                xytext=(elbow_k+1.5, wcss[elbow_k-1]+2000),
                arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2),
                color='#ffd93d', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='600')
    ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=13, fontweight='600')
    ax.set_title('Elbow Method for Optimal K', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(list(K_range))
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # WCSS table
    st.markdown("### 📋 WCSS Values")
    
    wcss_df = pd.DataFrame({
        'K': list(K_range),
        'WCSS': [f"{w:.2f}" for w in wcss],
        'Difference': ['-'] + [f"{wcss[i-1]-wcss[i]:.2f}" for i in range(1, len(wcss))]
    })
    
    st.dataframe(wcss_df.style.background_gradient(cmap='YlOrRd', subset=['WCSS']), 
                use_container_width=True, height=400)
    
    # Recommendation card
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #00f5d4; margin-top: 20px;">
        <h4 style="color: #00f5d4; margin-bottom: 10px;">💡 Recommendation</h4>
        <p style="color: #ccd6f6; line-height: 1.6;">
            Based on the elbow curve, <strong style="color: #00f5d4;">K={optimal_k}</strong> is selected. 
            The "elbow" occurs where WCSS reduction rate significantly slows — indicating diminishing returns 
            from adding more clusters. This balances model simplicity with segmentation quality.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# CLUSTERING
# ============================================
elif page == "🎯 Clustering":
    st.markdown("""
    <div style="padding: 20px 0 30px 0;">
        <h1 style="display: inline; background: linear-gradient(135deg, #ff6b6b, #ffd93d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎯 Clustering Results</h1>
        <p style="color: #8892b0; margin-top: 10px;">K-Means segmentation with {k} clusters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Cluster names and colors
    cluster_names = {
        0: "🛡️ Careful Spenders",
        1: "⭐ Standard Group",
        2: "🎯 VIP Targets",
        3: "💸 Careless Buyers",
        4: "🧠 Sensible Shoppers",
        5: "💎 Premium Elite",
        6: "🎪 Budget Hunters",
        7: "🌟 Rising Stars",
        8: "🔥 Hot Prospects",
        9: "💤 Dormant Accounts"
    }
    
    colors = ['#ff6b6b', '#00f5d4', '#ffd93d', '#c77dff', '#4d96ff', 
              '#6bcb77', '#ff9f43', '#ee5a24', '#009432', '#0652dd']
    
    # Main scatter plot
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    fig, ax = create_dark_figure(figsize=(16, 10))
    
    for i in range(optimal_k):
        mask = clusters == i
        cluster_data = X[mask]
        
        # Scatter with glow effect
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                  s=150, c=colors[i], alpha=0.85,
                  label=f'{cluster_names.get(i, f"Cluster {i}")} ({len(cluster_data)})',
                  edgecolors='white', linewidth=0.8, zorder=3)
        
        # Add subtle glow
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                  s=300, c=colors[i], alpha=0.1, zorder=2)
    
    # Centroids with star marker
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              s=500, c='white', marker='*', 
              edgecolors='black', linewidth=2, 
              label='Centroids', zorder=10)
    
    # Centroid labels
    for i, (x, y) in enumerate(centroids):
        ax.annotate(f'C{i}', (x, y), xytext=(5, 5), 
                   textcoords='offset points', color='white',
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Annual Income (k$)', fontsize=14, fontweight='600')
    ax.set_ylabel('Spending Score (1-100)', fontsize=14, fontweight='600')
    ax.set_title(f'Customer Segments (K={optimal_k})', fontsize=20, fontweight='bold', pad=20)
    ax.legend(facecolor='#0f0f23', edgecolor='#00f5d4', labelcolor='white', 
             fontsize=11, loc='upper left', framealpha=0.9)
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cluster detail cards
    st.markdown("### 📊 Segment Profiles")
    
    cols = st.columns(min(optimal_k, 5))
    
    for i, col in enumerate(cols):
        if i >= optimal_k:
            break
            
        cluster_data = df[df['Cluster'] == i]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_score = cluster_data['Spending Score (1-100)'].mean()
        avg_age = cluster_data['Age'].mean()
        
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid {colors[i]}; text-align: center;">
                <h4 style="color: {colors[i]}; margin-bottom: 15px; font-size: 1.1em;">{cluster_names.get(i, f'Cluster {i}')}</h4>
                <p style="color: white; font-size: 2.5em; font-weight: 800; margin: 0;">{len(cluster_data)}</p>
                <p style="color: #8892b0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">customers</p>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
                <p style="color: #ccd6f6; font-size: 13px; margin: 5px 0;">💰 Avg Income: <strong style="color: {colors[i]};">${avg_income:.0f}k</strong></p>
                <p style="color: #ccd6f6; font-size: 13px; margin: 5px 0;">⭐ Avg Score: <strong style="color: {colors[i]};">{avg_score:.0f}</strong></p>
                <p style="color: #ccd6f6; font-size: 13px; margin: 5px 0;">🎂 Avg Age: <strong style="color: {colors[i]};">{avg_age:.0f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# RESULTS
# ============================================
elif page == "📋 Results":
    st.markdown("""
    <div style="padding: 20px 0 30px 0;">
        <h1 style="display: inline; background: linear-gradient(135deg, #4d96ff, #00f5d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📋 Results & Export</h1>
        <p style="color: #8892b0; margin-top: 10px;">Segmented customer data ready for download</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    segment_map = {
        0: "Careful Spenders", 1: "Standard Group", 2: "VIP Targets",
        3: "Careless Buyers", 4: "Sensible Shoppers", 5: "Premium Elite",
        6: "Budget Hunters", 7: "Rising Stars", 8: "Hot Prospects", 9: "Dormant Accounts"
    }
    df['Segment'] = df['Cluster'].map(lambda x: segment_map.get(x, f'Cluster {x}'))
    
    # Data table
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00f5d4; margin-bottom: 20px;'>📋 Segmented Customer Records</h3>", unsafe_allow_html=True)
    
    styled_df = df.style.background_gradient(cmap='viridis', subset=['Cluster'])
    st.dataframe(styled_df, use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download section
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <p style="color: #8892b0; margin-bottom: 15px;">Download the complete segmented dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    csv = df.to_csv(index=False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='Customer_Segments.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    # Summary statistics
    st.markdown("### 📊 Segment Summary")
    
    summary = df.groupby('Segment').agg({
        'CustomerID': 'count',
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean'
    }).round(1)
    summary.columns = ['Count', 'Avg Age', 'Avg Income ($k)', 'Avg Score']
    summary = summary.sort_values('Count', ascending=False)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(
        summary.style.background_gradient(cmap='YlOrRd', subset=['Count'])
                  .background_gradient(cmap='viridis', subset=['Avg Income ($k)'])
                  .background_gradient(cmap='plasma', subset=['Avg Score']),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Business insights
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #ffd93d; margin-top: 20px;">
        <h4 style="color: #ffd93d; margin-bottom: 15px;">💡 Business Insights</h4>
        <ul style="color: #ccd6f6; line-height: 2;">
            <li><strong style="color: #00f5d4;">VIP Targets</strong> — High income + High spending. Focus loyalty programs here.</li>
            <li><strong style="color: #c77dff;">Careful Spenders</strong> — Low income + Low spending. Offer budget-friendly deals.</li>
            <li><strong style="color: #ff6b6b;">Careless Buyers</strong> — Low income + High spending. Risk group — monitor carefully.</li>
            <li><strong style="color: #4d96ff;">Sensible Shoppers</strong> — High income + Low spending. Upselling opportunity.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)