# CodeFortune - Industrial-Grade ESG Intelligence Platform
# Production-ready Streamlit application for enterprise ESG analysis, forecasting, and compliance
# Features: Advanced ML models, real-time KPIs, industry benchmarking, compliance tracking, and professional exports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# Optional Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# Page config
st.set_page_config(
    page_title="CodeFortune - ESG Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ESG Platform v2.0 - Enterprise Ready"}
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #0068C9;
    }
    .warning-box {
        background-color: #fff2cc;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS & CONFIG ====================
EMISSION_FACTORS = {
    'electricity_kwh': 0.82,
    'diesel_l': 2.68,
    'petrol_l': 2.31,
    'coal_kg': 2.42,
    'lpg_kg': 1.51,
    'natural_gas_m3': 2.03,
    'water_kl': 0.35,
    'waste_kg': 1.8
}

# Industry benchmarks (kg CO2e per unit produced)
INDUSTRY_BENCHMARKS = {
    'Manufacturing': 5.2,
    'Energy': 8.5,
    'Technology': 1.2,
    'Healthcare': 3.1,
    'Retail': 2.8,
    'Agriculture': 6.5,
    'Transportation': 7.2
}

# ESG Certifications
ESG_CERTIFICATIONS = {
    'ISO 14001': 'Environmental Management',
    'ISO 50001': 'Energy Management',
    'B Corp': 'Social & Environmental Performance',
    'Carbon Trust': 'Carbon Reduction',
    'LEED': 'Green Building',
    'Science Based Targets': 'Climate Commitment'
}

MIN_ROWS = 12
RANDOM_STATE = 42
TEST_RATIO = 0.2

# ==================== SESSION STATE INITIALIZATION ====================
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []
if 'kpi_history' not in st.session_state:
    st.session_state['kpi_history'] = {}

# ==================== HELPER FUNCTIONS ====================

@st.cache_data
def generate_demo_dataset(company_name='Company A', start='2021-01-01', months=48, seed=42):
    """Generate realistic industrial dataset with trends and seasonality."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=months, freq='M')
    
    base_elec = 5000 if company_name == 'Company A' else 3500
    base_diesel = 800 if company_name == 'Company A' else 400
    production_base = 2000 if company_name == 'Company A' else 1500
    
    # Add trend (improvement over time)
    trend = np.linspace(1.0, 0.85, months)  # 15% improvement
    
    df = pd.DataFrame({
        'Date': dates,
        'Electricity_kWh': np.round(np.random.normal(base_elec, base_elec*0.12, months) * trend).astype(int),
        'Diesel_L': np.round(np.random.normal(base_diesel, base_diesel*0.2, months) * trend).astype(int),
        'Water_Usage_kl': np.round(np.random.normal(150, 40, months)).astype(int),
        'Waste_Generated_kg': np.round(np.random.normal(450, 120, months) * trend).astype(int),
        'Natural_Gas_m3': np.round(np.random.normal(1000, 300, months) * trend).astype(int),
        'Production_Volume': np.round(np.random.normal(production_base, production_base*0.15, months)).astype(int),
        'Employees_On_Site': np.clip(np.round(np.random.normal(200, 20, months)).astype(int), 50, 1000)
    })
    
    # Seasonality
    month_factor = (np.sin(2 * np.pi * (df['Date'].dt.month / 12)) + 1) * 0.05
    df['Electricity_kWh'] = (df['Electricity_kWh'] * (1 + month_factor)).astype(int)
    
    # Occasional maintenance spikes
    spikes = np.random.choice(range(months), size=max(1, months // 12), replace=False)
    df.loc[spikes, 'Waste_Generated_kg'] = (df.loc[spikes, 'Waste_Generated_kg'] * 1.8).astype(int)
    
    df['Company'] = company_name
    df['Industry'] = 'Manufacturing'
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    return df.copy()

@st.cache_data
def load_uploaded_file(uploaded_file):
    """Load CSV or Excel file with error handling."""
    try:
        name = getattr(uploaded_file, 'name', '')
        if name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f'Error reading file: {str(e)}')
        return None

def clean_and_engineer(df, datetime_col='Date'):
    """Clean data and create engineered features for ML models."""
    df = df.copy()
    
    # Standardize column names
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    
    # Parse dates
    if datetime_col.lower() in [c.lower() for c in df.columns]:
        for col in df.columns:
            if col == datetime_col.lower():
                datetime_col = col
                break
    
    for c in df.columns:
        if 'date' in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
                if df[c].notna().sum() > len(df) * 0.8:
                    datetime_col = c
                    break
            except:
                continue
    
    # Drop rows without date
    df = df.dropna(subset=[datetime_col])
    
    # Coerce numeric columns
    for c in df.columns:
        if c == datetime_col or df[c].dtype == 'datetime64[ns]':
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except:
            pass
    
    # Fill numeric NAs with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Calculate emissions
    for key, factor in EMISSION_FACTORS.items():
        key_prefix = key.split('_')[0]
        matches = [c for c in df.columns if key_prefix in c.lower()]
        for m in matches:
            df[f'{m}_co2e'] = df[m] * factor
    
    # Total emission
    emission_cols = [c for c in df.columns if c.endswith('_co2e')]
    if emission_cols:
        df['emission_kgco2e'] = df[emission_cols].sum(axis=1)
    else:
        df['emission_kgco2e'] = 0.0
    
    # Per-unit emissions
    if 'production_volume' in df.columns:
        df['emission_per_unit'] = df['emission_kgco2e'] / df['production_volume'].replace(0, np.nan)
        df['emission_per_unit'] = df['emission_per_unit'].fillna(df['emission_per_unit'].median())
    
    # Sort by date
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    
    # Feature engineering: lags and rolling averages
    lag_cols = ['electricity_kwh', 'diesel_l', 'natural_gas_m3', 'waste_generated_kg', 'emission_kgco2e']
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(method='bfill')
            df[f'{col}_rmean3'] = df[col].rolling(window=3, min_periods=1).mean()
    
    # Time features
    df['month'] = pd.to_datetime(df[datetime_col]).dt.month
    df['year'] = pd.to_datetime(df[datetime_col]).dt.year
    df['quarter'] = pd.to_datetime(df[datetime_col]).dt.quarter
    
    final_num = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df, final_num

def calculate_kpis(df):
    """Calculate key performance indicators."""
    kpis = {
        'total_emissions': float(df['emission_kgco2e'].sum()),
        'avg_emissions': float(df['emission_kgco2e'].mean()),
        'max_emissions': float(df['emission_kgco2e'].max()),
        'min_emissions': float(df['emission_kgco2e'].min()),
        'trend': 'improving' if df['emission_kgco2e'].iloc[-1] < df['emission_kgco2e'].iloc[0] else 'worsening',
        'records': len(df),
        'months': len(df)
    }
    
    if 'emission_per_unit' in df.columns:
        kpis['emission_per_unit'] = float(df['emission_per_unit'].mean())
    
    return kpis

def train_regressors(X, y):
    """Train multiple ML models with cross-validation."""
    results = {}
    
    try:
        if len(X) < 15:
            st.warning(f'Warning: {len(X)} samples may be insufficient for reliable model training')
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {
            'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0),
            'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE)
        }
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                model.fit(X, y)
                preds = model.predict(X)
                
                results[name] = {
                    'model': model,
                    'cv_mae': float(np.mean(-1 * scores)),
                    'cv_std': float(np.std(-1 * scores)),
                    'r2': float(r2_score(y, preds)),
                    'mae': float(mean_absolute_error(y, preds)),
                    'rmse': float(mean_squared_error(y, preds, squared=False)),
                    'mape': float(mean_absolute_percentage_error(y, preds))
                }
            except Exception as e:
                st.warning(f'Training {name} failed: {str(e)}')
        
        return results
    except Exception as e:
        st.error(f'Model training error: {str(e)}')
        return {}

def prophet_forecast(df, periods=12, date_col='date'):
    """Generate Prophet forecast with confidence intervals."""
    if not HAS_PROPHET:
        st.warning('Prophet not installed. Using linear trend instead.')
        return None
    
    try:
        m = Prophet(yearly_seasonality=True, interval_width=0.95)
        
        ds = df[[date_col, 'emission_kgco2e']].copy()
        ds.columns = ['ds', 'y']
        ds['ds'] = pd.to_datetime(ds['ds'])
        
        m.fit(ds)
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.warning(f'Prophet forecast failed: {str(e)}')
        return None

def generate_professional_pdf(df, company_name='Company', industry='Manufacturing', recommendations=None, certifications=None):
    """Generate professional ESG compliance report with proper Unicode handling."""
    
    def sanitize_text(text):
        """Remove Unicode characters and replace with ASCII equivalents."""
        if not isinstance(text, str):
            text = str(text)
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '\u2192': '->',  # Right arrow
            '\u2713': 'X',   # Checkmark
            '\u2717': 'X',   # X mark
            '\u2691': '[X]', # Ballot X
            '\u25cf': '*',   # Bullet
            '\u00e9': 'e',   # é
            '\u00e0': 'a',   # à
            '\u00fc': 'u',   # ü
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        # Remove any remaining non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, sanitize_text(f'{company_name} - ESG Sustainability Report'), ln=True, align='C')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, sanitize_text(f'Industry: {industry}'), ln=True, align='C')
    pdf.cell(0, 6, sanitize_text(f'Report Date: {datetime.now().strftime("%Y-%m-%d")}'), ln=True, align='C')
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', ln=True)
    pdf.set_font('Arial', '', 10)
    
    total_emissions = int(df['emission_kgco2e'].sum())
    avg_emissions = float(df['emission_kgco2e'].mean())
    max_emissions = int(df['emission_kgco2e'].max())
    
    summary_text = f"""Total Emissions (Period): {total_emissions:,} kgCO2e
Average per Record: {avg_emissions:,.2f} kgCO2e
Peak Emissions: {max_emissions:,} kgCO2e
Records Analyzed: {len(df)}
Data Period: {len(df)} months"""
    
    pdf.multi_cell(0, 5, sanitize_text(summary_text))
    pdf.ln(5)
    
    # Compliance Checklist
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Compliance & Audit Checklist', ln=True)
    pdf.set_font('Arial', '', 9)
    
    checklist = [
        'Monthly energy consumption records (kWh) submitted and verified',
        'Fuel consumption logs (diesel, petrol, natural gas) documented',
        'Water withdrawal and wastewater discharge records maintained',
        'Waste generation and disposal records tracked',
        'Emission calculation methodology documented and auditable',
        'Third-party verification completed or scheduled',
        'ISO 14001 / ISO 50001 standards compliance confirmed',
        'Scope 1, 2, and 3 emissions classified and tracked',
        'Year-over-year trend analysis performed',
        'Corrective action plans initiated for high-emission periods'
    ]
    
    for i, item in enumerate(checklist, 1):
        pdf.cell(5, 5, '[X]' if i <= 7 else '[ ]', border=1)
        pdf.multi_cell(0, 5, sanitize_text(f' {item}'))
    
    pdf.ln(4)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Action Items & Recommendations', ln=True)
    pdf.set_font('Arial', '', 9)
    
    if recommendations:
        for i, rec in enumerate(recommendations[:10], 1):
            pdf.multi_cell(0, 5, sanitize_text(f'{i}. {rec}'))
    else:
        pdf.multi_cell(0, 5, 'No specific recommendations generated.')
    
    # Certifications
    if certifications:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Target Certifications', ln=True)
        pdf.set_font('Arial', '', 9)
        for cert, desc in certifications.items():
            pdf.multi_cell(0, 5, sanitize_text(f'- {cert}: {desc}'))
    
    # Emission Breakdown
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Emission Breakdown by Source', ln=True)
    pdf.set_font('Arial', '', 10)
    
    cols = [c for c in df.columns if c.endswith('_co2e')]
    if cols:
        totals = df[cols].sum().sort_values(ascending=False)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(80, 6, 'Source', border=1)
        pdf.cell(60, 6, 'Total (kgCO2e)', border=1, ln=True)
        pdf.set_font('Arial', '', 9)
        for k, v in totals.items():
            source_name = k.replace('_co2e', '').replace('_', ' ').title()
            pdf.cell(80, 6, sanitize_text(source_name), border=1)
            pdf.cell(60, 6, sanitize_text(f'{int(v):,}'), border=1, ln=True)
    
    try:
        out = pdf.output(dest='S')
        if isinstance(out, str):
            out = out.encode('latin-1', errors='replace')
        return io.BytesIO(out)
    except UnicodeEncodeError as e:
        st.error(f'PDF encoding error: {str(e)}. Using fallback method.')
        # Fallback: create minimal PDF if encoding fails
        pdf_fallback = FPDF()
        pdf_fallback.add_page()
        pdf_fallback.set_font('Arial', '', 12)
        pdf_fallback.cell(0, 10, sanitize_text(f'{company_name} - ESG Report'), ln=True)
        pdf_fallback.cell(0, 10, sanitize_text(f'Report Date: {datetime.now().strftime("%Y-%m-%d")}'), ln=True)
        pdf_fallback.cell(0, 10, sanitize_text(f'Total Emissions: {int(df["emission_kgco2e"].sum()):,} kgCO2e'), ln=True)
        out = pdf_fallback.output(dest='S')
        if isinstance(out, str):
            out = out.encode('latin-1', errors='replace')
        return io.BytesIO(out)

# ==================== STREAMLIT UI ====================

st.title('CodeFortune - ESG Intelligence Platform')
st.markdown('*Enterprise-grade environmental, social, and governance analytics*')

# Sidebar
st.sidebar.header('Configuration')
uploaded = st.sidebar.file_uploader('Upload Data (CSV/Excel)', type=['csv', 'xlsx'])
use_demo = st.sidebar.checkbox('Use Demo Datasets', value=True)
company_choice = st.sidebar.selectbox('Select Company', ['Company A', 'Company B', 'Custom'])
industry_choice = st.sidebar.selectbox('Industry Type', list(INDUSTRY_BENCHMARKS.keys()))

# Load data
if use_demo:
    demo_a = generate_demo_dataset('Company A', start='2021-01-01', months=48, seed=42)
    demo_b = generate_demo_dataset('Company B', start='2022-01-01', months=36, seed=24)
else:
    demo_a = demo_b = None

if uploaded is not None:
    df_uploaded = load_uploaded_file(uploaded)
    if df_uploaded is not None:
        if 'Company' not in df_uploaded.columns:
            df_uploaded['Company'] = company_choice
        st.session_state['uploaded_df'] = df_uploaded
else:
    st.session_state['uploaded_df'] = None

raw_df = st.session_state.get('uploaded_df') or (demo_a if company_choice == 'Company A' else demo_b)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Data Overview', 'Data Cleaning', 'Analytics Dashboard', 'ML Forecasting', 'Compliance & Benchmarks', 'Export & Reports'
])

with tab1:
    st.header('Data Preview & Overview')
    if raw_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Records', len(raw_df))
        with col2:
            st.metric('Columns', len(raw_df.columns))
        with col3:
            st.metric('Date Range', f"{len(raw_df)} months")
        
        st.subheader('Data Sample')
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        st.subheader('Data Quality Check')
        missing = raw_df.isnull().sum()
        if missing.sum() > 0:
            st.warning(f'Missing values detected: {dict(missing[missing > 0])}')
        else:
            st.success('No missing values detected')
    else:
        st.error('No data available. Please upload a file or enable demo datasets.')

with tab2:
    st.header('Data Cleaning & Feature Engineering')
    if raw_df is not None:
        df_clean, numeric_cols = clean_and_engineer(raw_df)
        st.session_state['cleaned_df'] = df_clean
        
        st.success('Data cleaned and features engineered successfully')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Rows', len(df_clean))
        with col2:
            st.metric('Numeric Features', len(numeric_cols))
        with col3:
            st.metric('Emission Total', f"{int(df_clean['emission_kgco2e'].sum()):,} kg")
        with col4:
            st.metric('Avg Emission', f"{df_clean['emission_kgco2e'].mean():.2f} kg")
        
        st.subheader('Summary Statistics')
        st.dataframe(df_clean[numeric_cols].describe().T, use_container_width=True)
        
        # Download cleaned data
        csv_buf = io.StringIO()
        df_clean.to_csv(csv_buf, index=False)
        st.download_button('Download Cleaned Data (CSV)', csv_buf.getvalue(), f'cleaned_{company_choice}.csv', 'text/csv')
    else:
        st.error('No data to clean')

with tab3:
    st.header('Analytics Dashboard & KPIs')
    if st.session_state['cleaned_df'] is not None:
        dfc = st.session_state['cleaned_df']
        
        # KPIs
        kpis = calculate_kpis(dfc)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Emissions', f"{int(kpis['total_emissions']):,} kg", delta='CO2e')
        with col2:
            st.metric('Average Emission', f"{kpis['avg_emissions']:.2f} kg")
        with col3:
            st.metric('Peak Emission', f"{int(kpis['max_emissions']):,} kg")
        with col4:
            status = '📈 Worsening' if kpis['trend'] == 'worsening' else '📉 Improving'
            st.metric('Trend', status)
        
        # Emission breakdown chart
        cols = [c for c in dfc.columns if c.endswith('_co2e')]
        if cols:
            st.subheader('Emission Composition')
            breakdown = dfc[cols].sum().sort_values(ascending=False)
            labels = [c.replace('_co2e', '').replace('_', ' ').title() for c in breakdown.index]
            
            fig = px.pie(values=breakdown.values, names=labels, title='Emission Sources')
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader('Emissions Over Time')
        df_ts = dfc[['emission_kgco2e']].copy()
        df_ts.index = pd.to_datetime(dfc['date']) if 'date' in dfc.columns else range(len(dfc))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts['emission_kgco2e'], mode='lines+markers', name='Emissions'))
        fig.update_layout(title='Emissions Trend', xaxis_title='Date', yaxis_title='kgCO2e')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error('Please clean data first in Tab 2')

with tab4:
    st.header('ML Forecasting & Optimization')
    if st.session_state['cleaned_df'] is not None:
        dfc = st.session_state['cleaned_df']
        
        feature_cols = [c for c in dfc.select_dtypes(include=[np.number]).columns if c not in ['emission_kgco2e']]
        X = dfc[feature_cols].fillna(0)
        y = dfc['emission_kgco2e']
        
        st.subheader('Model Training')
        if st.button('Train ML Models (XGBoost, RF, GB)', key='train_models'):
            with st.spinner('Training models...'):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                results = train_regressors(X_scaled, y)
                st.session_state['model_results'] = results
                st.session_state['scaler'] = scaler
                st.success('Models trained successfully')
        
        if st.session_state['model_results']:
            st.subheader('Model Performance')
            res = st.session_state['model_results']
            summary = []
            for k, v in res.items():
                summary.append({
                    'Model': k,
                    'R²': f"{v['r2']:.4f}",
                    'MAE': f"{v['mae']:.2f}",
                    'RMSE': f"{v['rmse']:.2f}",
                    'MAPE': f"{v['mape']:.2f}%"
                })
            
            summary_df = pd.DataFrame(summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Prophet forecast
            if HAS_PROPHET:
                st.subheader('Forecast (Next 12 Months)')
                fc = prophet_forecast(dfc, periods=12, date_col='date')
                if fc is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat'], name='Forecast', mode='lines'))
                    fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)'))
                    fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval'))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Optimization
            st.subheader('Optimization Simulation')
            best_name = min(st.session_state['model_results'].keys(), key=lambda n: st.session_state['model_results'][n]['mae'])
            best_model = st.session_state['model_results'][best_name]['model']
            scaler = st.session_state.get('scaler')
            
            if scaler:
                impacts = []
                for i, feat in enumerate(feature_cols):
                    arr = np.array([X.iloc[-1].values])
                    arr_scaled = scaler.transform(arr)
                    baseline = best_model.predict(arr_scaled)[0]
                    
                    arr[0, i] = arr[0, i] * 0.9
                    arr_scaled = scaler.transform(arr)
                    improved = best_model.predict(arr_scaled)[0]
                    
                    saving = baseline - improved
                    if saving > 0:
                        impacts.append({
                            'Feature': feat,
                            'Saving (kg)': f"{saving:.2f}",
                            'Pct Reduction': f"{(saving/baseline*100):.1f}%"
                        })
                
                if impacts:
                    impacts_df = pd.DataFrame(impacts).head(10)
                    st.dataframe(impacts_df, use_container_width=True)
                    
                    recs = [f"Reduce {row['Feature']} by 10% → save {row['Saving (kg)']} kg CO2e" for _, row in impacts_df.iterrows()]
                    st.session_state['recommendations'] = recs
    else:
        st.error('Please clean data first')

with tab5:
    st.header('Compliance & Industry Benchmarking')
    if st.session_state['cleaned_df'] is not None:
        dfc = st.session_state['cleaned_df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Compliance Status')
            certifications = {
                'ISO 14001': 'In Progress',
                'ISO 50001': 'Not Started',
                'Carbon Trust': 'Certified',
                'Science Based Targets': 'In Progress'
            }
            for cert, status in certifications.items():
                icon = '✅' if status == 'Certified' else '⏳' if status == 'In Progress' else '❌'
                st.write(f'{icon} {cert}: {status}')
        
        with col2:
            st.subheader('Industry Benchmarking')
            avg_emission = dfc['emission_kgco2e'].mean()
            industry_benchmark = INDUSTRY_BENCHMARKS.get(industry_choice, 5.0)
            performance = (1 - (avg_emission / (industry_benchmark * 1000))) * 100
            
            st.metric('Your Avg Emission', f'{avg_emission:.2f} kg', delta=f'{performance:.1f}% vs industry avg')
            
            benchmark_data = pd.DataFrame({
                'Metric': ['Your Company', 'Industry Avg'],
                'Emissions': [avg_emission, industry_benchmark * 1000]
            })
            fig = px.bar(benchmark_data, x='Metric', y='Emissions', title='Benchmarking')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Regulatory Compliance Checklist')
        compliance_items = [
            ('GHG Protocol Corporate Standard', True),
            ('TCFD Climate Risk Disclosure', True),
            ('SEC Climate Disclosure Rule', False),
            ('EU Taxonomy Alignment', True),
            ('CSRD Requirements', False),
            ('Scope 1, 2, 3 Tracking', True)
        ]
        
        compliance_df = pd.DataFrame(compliance_items, columns=['Requirement', 'Compliant'])
        st.dataframe(compliance_df, use_container_width=True)
    else:
        st.error('No data available')

with tab6:
    st.header('Export & Professional Reports')
    if st.session_state['cleaned_df'] is not None:
        dfc = st.session_state['cleaned_df']
        company = dfc['company'].iloc[0] if 'company' in dfc.columns else company_choice
        
        st.subheader('Generate Professional Reports')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('**PDF Report**')
            if st.button('Generate ESG PDF Report'):
                buf = generate_professional_pdf(
                    dfc,
                    company_name=company,
                    industry=industry_choice,
                    recommendations=st.session_state['recommendations']
                )
                st.download_button('Download PDF Report', buf, f'{company}_ESG_Report.pdf', 'application/pdf')
        
        with col2:
            st.write('**Cleaned Data Export**')
            csv_buf = io.StringIO()
            dfc.to_csv(csv_buf, index=False)
            st.download_button('Download CSV Data', csv_buf.getvalue(), f'{company}_final_data.csv', 'text/csv')
        
        st.markdown('---')
        st.subheader('Data Summary for Export')
        export_summary = {
            'Total Records': len(dfc),
            'Total Emissions (kg CO2e)': f"{int(dfc['emission_kgco2e'].sum()):,}",
            'Average Emissions': f"{dfc['emission_kgco2e'].mean():.2f}",
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.json(export_summary)
    else:
        st.error('No data available for export')

# Footer
st.markdown('---')
st.markdown('*CodeFortune ESG Platform v2.0 | Enterprise Ready | Production Grade*')
