import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.io as pio

# 1. Page Configuration
st.set_page_config(
    layout='wide',
    page_title='Zomato Analytics 🍽️',
    page_icon='🔥',
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 50px;
        color: #E23744;
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 25px;
        color: #2D2D2D;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Data (THE SOLUTION IS HERE)
@st.cache_data
def load_data():
    try:
        # Reading the compressed Parquet file instead of CSV
        df = pd.read_parquet("cleaned_df.parquet")
    except:
        # Fallback if parquet is not found (for local testing with csv)
        df = pd.read_csv("cleaned_df.csv")
    
    # Drop unnecessary index column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

df = load_data()

# 4. Sidebar Navigation & Filters
st.sidebar.image("https://b.zmtcdn.com/web_assets/b40b97e677bc7b2ca77c58c61db266fe1603954218.png", width=200)
st.sidebar.markdown("---")
page = st.sidebar.radio('🚀 Navigation', ['Home', 'Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])

st.sidebar.markdown("---")
st.sidebar.header("Filter Data 🔍")

# Location Filter
# Check if dataframe is not empty to avoid errors
if not df.empty:
    selected_location = st.sidebar.multiselect("Select Location:", options=df['location'].unique(), default=df['location'].unique()[:5])

    if selected_location:
        df_filtered = df[df['location'].isin(selected_location)]
    else:
        df_filtered = df.copy()
else:
    st.error("Data could not be loaded. Please check the file.")
    st.stop()

# ==========================================
# PAGE 1: HOME (Dashboard Overview)
# ==========================================
if page == 'Home':
    st.markdown('<p class="main-header">Zomato Bangalore Restaurants 📊</p>', unsafe_allow_html=True)
    st.image('https://b.zmtcdn.com/web_assets/81f3ff974d82520780078ba1cfbd453a1583259680.png', use_column_width=True)
    
    st.write("### 📈 Key Performance Indicators (KPIs)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Restaurants", f"{df_filtered.shape[0]:,}")
    with c2:
        st.metric("Avg Rating ⭐", f"{df_filtered['rate'].mean():.2f}/5")
    with c3:
        st.metric("Avg Cost (2 People) 💰", f"{df_filtered['cost'].mean():.0f} INR")
    with c4:
        st.metric("Online Delivery 🛵", f"{df_filtered[df_filtered['online_order']=='Yes'].shape[0]}")

    st.markdown("---")
    st.write("### 🗂️ Data Preview")
    with st.expander("Show Raw Data"):
        st.dataframe(df_filtered.head(100), use_container_width=True)

# ==========================================
# PAGE 2: UNIVARIATE ANALYSIS
# ==========================================
elif page == 'Univariate Analysis':
    st.markdown('<h1 style="color:#E23744;">Univariate Analysis 🧐</h1>', unsafe_allow_html=True)
    
    col_to_plot = st.selectbox("Select Column to Visualize:", ['rate', 'cost', 'online_order', 'book_table', 'rest_type'])
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        if df_filtered[col_to_plot].dtype == 'object' or len(df_filtered[col_to_plot].unique()) < 20:
            fig = px.pie(df_filtered, names=col_to_plot, title=f'Distribution of {col_to_plot}', hole=0.4, template="plotly_dark")
        else:
            fig = px.histogram(df_filtered, x=col_to_plot, title=f'Distribution of {col_to_plot}', nbins=30, template="plotly_dark", color_discrete_sequence=['#E23744'])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.write(f"### Stats for {col_to_plot}")
        if df_filtered[col_to_plot].dtype != 'object':
            st.write(df_filtered[col_to_plot].describe())
        else:
            st.write(df_filtered[col_to_plot].value_counts())

# ==========================================
# PAGE 3: BIVARIATE ANALYSIS
# ==========================================
elif page == 'Bivariate Analysis':
    st.markdown('<h1 style="color:#E23744;">Bivariate Analysis ⚖️</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💰 Price vs Rating", "📍 Locations", "📚 Booking Effect"])
    
    with tab1:
        st.subheader("Does Price Affect Quality?")
        fig_scatter = px.scatter(df_filtered, x='cost', y='rate', color='online_order', 
                                 title='Cost vs Rate Correlation', opacity=0.6, template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with tab2:
        st.subheader("Top Locations Count")
        top_loc = df_filtered['location'].value_counts().head(10).reset_index()
        top_loc.columns = ['location', 'count']
        fig_bar = px.bar(top_loc, x='count', y='location', orientation='h', title="Top 10 Locations", 
                         text='count', color='count', template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Table Booking vs Rating")
        fig_box = px.box(df_filtered, x='book_table', y='rate', color='book_table', 
                         title="Do restaurants with Table Booking have higher ratings?", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PAGE 4: MULTIVARIATE ANALYSIS
# ==========================================
elif page == 'Multivariate Analysis':
    st.markdown('<h1 style="color:#E23744;">Multivariate Analysis 🧊</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Correlation Heatmap")
        
        numeric_df = df_filtered.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col2:
        st.subheader("🌳 Cost Hierarchy (Treemap)")
        
        if not df_filtered.empty:
            treemap_df = df_filtered.groupby(['location', 'rest_type'])['cost'].mean().reset_index()
            
            top_locations_list = df_filtered['location'].value_counts().head(10).index
            treemap_df = treemap_df[treemap_df['location'].isin(top_locations_list)]
            
            fig_tree = px.treemap(treemap_df, path=['location', 'rest_type'], values='cost',
                                  color='cost', color_continuous_scale='Magma', template="plotly_dark")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.warning("Please select at least one location from the sidebar.")
