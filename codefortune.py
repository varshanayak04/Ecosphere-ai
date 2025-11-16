# CodeFortune_ESG_App_streamlit_fixed_full.py
"""
Full final: Hardened & optimized single-file Streamlit app for CodeFortune ESG Intelligence.
Includes:
 - Robust data loading & cleaning
 - train_regressors() training multiple models and producing forecasts
 - generate_professional_pdf_reportlab() (Option D) — custom order, no charts, no page breaks
 - Tabs 4/5/6 replaced to integrate ML forecast and PDF generation
"""
import streamlit as st
st.set_page_config(page_title="Eco Sphere AI- ESG Intelligence", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, os, tempfile, traceback
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libs
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Optional libs
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ReportLab for PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ------------------ CONFIG ------------------
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

INDUSTRY_BENCHMARKS = {
    'manufacturing': 5.2,
    'energy': 8.5,
    'technology': 1.2,
    'healthcare': 3.1,
    'retail': 2.8,
    'agriculture': 6.5,
    'transportation': 7.2
}

ESG_CERTIFICATIONS = {
    'ISO 14001': 'Environmental Management',
    'ISO 50001': 'Energy Management',
    'B Corp': 'Social & Environmental Performance',
    'Carbon Trust': 'Carbon Reduction',
    'LEED': 'Green Building',
    'Science Based Targets': 'Climate Commitment'
}

RANDOM_STATE = 42

# ------------------ SESSION DEFAULTS ------------------
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []
if 'ml_forecast' not in st.session_state:
    st.session_state['ml_forecast'] = None
if 'ml_recommendations' not in st.session_state:
    st.session_state['ml_recommendations'] = []

# ------------------ HELPERS ------------------
@st.cache_data
def generate_demo_dataset(company_name='Company A', start='2016-01-01', months=120, seed=42, industry='manufacturing'):
    """Synthetic demo dataset generator (deterministic via seed)."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=months, freq='M')

    base_elec = 5000 if company_name == 'Company A' else 3500
    base_diesel = 800 if company_name == 'Company A' else 400
    production_base = 2000 if company_name == 'Company A' else 1500

    trend = np.linspace(1.0, 0.85, months)

    df = pd.DataFrame({
        'date': dates,
        'electricity_kwh': np.round(np.random.normal(base_elec, base_elec*0.12, months) * trend).astype(int),
        'diesel_l': np.round(np.random.normal(base_diesel, base_diesel*0.2, months) * trend).astype(int),
        'water_kl': np.round(np.random.normal(150, 40, months)).astype(int),
        'waste_kg': np.round(np.random.normal(450, 120, months) * trend).astype(int),
        'natural_gas_m3': np.round(np.random.normal(1000, 300, months) * trend).astype(int),
        'production_volume': np.round(np.random.normal(production_base, production_base*0.15, months)).astype(int),
        'employees_on_site': np.clip(np.round(np.random.normal(200, 20, months)).astype(int), 50, 1000)
    })

    month_factor = (np.sin(2 * np.pi * (df['date'].dt.month / 12)) + 1) * 0.05
    df['electricity_kwh'] = (df['electricity_kwh'] * (1 + month_factor)).astype(int)

    spikes = np.random.choice(range(months), size=max(1, months // 12), replace=False)
    df.loc[spikes, 'waste_kg'] = (df.loc[spikes, 'waste_kg'] * 1.8).astype(int)

    df['company'] = company_name
    df['industry'] = industry.lower()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    return df.copy()

def sanitize_text(text):
    """Small sanitizer for text used in PDF & UI."""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    replacements = {'→': '->', '✓': 'X', '✗': 'X'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def safe_read_file(uploaded_file):
    """
    Robust file reader: returns DataFrame or raises descriptive Exception.
    Handles CSV and Excel. Resets file stream if necessary.
    """
    try:
        name = getattr(uploaded_file, 'name', '')
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if name and name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            try:
                df = pd.read_csv(uploaded_file, dtype=str)
            except Exception:
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                df = pd.read_excel(uploaded_file, engine='openpyxl' if str(uploaded_file).endswith('xlsx') else None)
        return df
    except Exception as e:
        raise RuntimeError(f'Could not read file: {e}')

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df

def validate_uploaded_df(df):
    if df is None:
        return False, 'No dataframe provided'
    if not isinstance(df, pd.DataFrame):
        return False, 'Uploaded object is not a DataFrame'
    if df.shape[0] == 0:
        return False, 'Uploaded file contains 0 rows'
    if df.shape[1] == 0:
        return False, 'Uploaded file contains 0 columns'
    return True, 'OK'

def load_uploaded_file(uploaded_file, default_industry='manufacturing', default_company='Uploaded Company'):
    try:
        df = safe_read_file(uploaded_file)
        df = normalize_columns(df)

        if 'industry' not in df.columns:
            df['industry'] = default_industry.lower() if default_industry else 'unknown'
        if 'company' not in df.columns:
            df['company'] = default_company if default_company else 'Uploaded Company'
        ok, msg = validate_uploaded_df(df)
        if not ok:
            st.error(f'Uploaded file validation failed: {msg}')
            return None
        return df
    except Exception as e:
        st.error(f'Error reading file: {e}')
        return None

def clean_and_engineer(df):
    if df is None:
        return None, []

    df = df.copy()
    df = normalize_columns(df)

    if 'industry' not in df.columns:
        df['industry'] = 'unknown'
    if 'company' not in df.columns:
        df['company'] = 'unknown'

    date_cols = [c for c in df.columns if 'date' in c or 'month' in c]
    date_col = None
    for c in date_cols:
        try:
            tmp = pd.to_datetime(df[c], errors='coerce')
            if tmp.notna().sum() > 0:
                date_col = c
                df[c] = tmp
                break
        except Exception:
            continue

    if date_col is None:
        df = df.reset_index(drop=True)
        df['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(df.index, unit='M')
    else:
        df.rename(columns={date_col: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).reset_index(drop=True)

    for c in df.columns:
        if c == 'date' or np.issubdtype(df[c].dtype, np.datetime64):
            continue
        try:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','').str.strip(), errors='coerce')
        except Exception:
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        try:
            med = df[col].median()
            if np.isnan(med):
                med = 0
        except Exception:
            med = 0
        df[col] = df[col].fillna(med)

    for canonical, factor in EMISSION_FACTORS.items():
        key = canonical.split('_')[0]
        matches = [c for c in df.columns if key in c]
        for m in matches:
            try:
                df[f'{m}_co2e'] = df[m] * factor
            except Exception:
                df[f'{m}_co2e'] = 0.0

    emission_cols = [c for c in df.columns if c.endswith('_co2e')]
    if emission_cols:
        df['emission_kgco2e'] = df[emission_cols].sum(axis=1)
    else:
        if 'emission' in df.columns:
            df['emission_kgco2e'] = df['emission']
        else:
            df['emission_kgco2e'] = 0.0

    if 'production_volume' in df.columns:
        df['emission_per_unit'] = df['emission_kgco2e'] / df['production_volume'].replace(0, np.nan)
        if df['emission_per_unit'].isna().all():
            df['emission_per_unit'] = df['emission_per_unit'].fillna(0)
        else:
            df['emission_per_unit'] = df['emission_per_unit'].fillna(df['emission_per_unit'].median())

    df = df.sort_values(by='date').reset_index(drop=True)

    for col in ['electricity_kwh', 'diesel_l', 'natural_gas_m3', 'waste_kg', 'emission_kgco2e']:
        if col in df.columns:
            for lag in [1,2,3]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(method='bfill')
            df[f'{col}_rmean3'] = df[col].rolling(window=3, min_periods=1).mean().fillna(method='bfill')

    cols = df.columns.tolist()
    for c in ['industry','company']:
        if c in cols:
            cols.remove(c)
            cols.insert(0, c)
    if 'date' in cols:
        cols.remove('date')
        cols.insert(2, 'date')
    df = df[cols]

    final_num = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, final_num

def calculate_kpis(df):
    kpis = {}
    if df is None or len(df)==0:
        kpis['total_emissions'] = 0.0
        kpis['avg_emissions'] = 0.0
        kpis['max_emissions'] = 0.0
        kpis['min_emissions'] = 0.0
        kpis['trend'] = 'stable'
        kpis['records'] = 0
        kpis['emission_per_unit'] = 0.0
        return kpis

    kpis['total_emissions'] = float(df['emission_kgco2e'].sum()) if 'emission_kgco2e' in df.columns else 0.0
    kpis['avg_emissions'] = float(df['emission_kgco2e'].mean()) if 'emission_kgco2e' in df.columns else 0.0
    kpis['max_emissions'] = float(df['emission_kgco2e'].max()) if 'emission_kgco2e' in df.columns else 0.0
    kpis['min_emissions'] = float(df['emission_kgco2e'].min()) if 'emission_kgco2e' in df.columns else 0.0
    try:
        kpis['trend'] = 'improving' if df['emission_kgco2e'].iloc[-1] < df['emission_kgco2e'].iloc[0] else 'worsening'
    except Exception:
        kpis['trend'] = 'stable'
    kpis['records'] = len(df)
    if 'emission_per_unit' in df.columns:
        kpis['emission_per_unit'] = float(df['emission_per_unit'].mean())
    else:
        kpis['emission_per_unit'] = 0.0
    return kpis

# -------------------------------
# GLOBAL: save_fig_to_file()
# -------------------------------
def save_fig_to_file(fig, dpi=150):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
    return path

# -------------------------------
# 1) train_regressors()
# -------------------------------
def train_regressors(X, y, use_xgb=HAS_XGB):
    """
    Train regressors and return a dict of results (metrics + model + in-sample preds).
    Returns:
      results: {model_name: {'model':model,'preds':array,'cv_mae':..., 'r2':..., 'mae':..., 'rmse':..., 'mape':...}}
    """
    results = {}
    try:
        X_arr = X.values if hasattr(X, "values") else np.array(X)
        y_arr = np.array(y)

        n = len(y_arr)
        if n < 8:
            st.warning(f"Warning: {n} samples — models may overfit or metrics may be unreliable.")

        try:
            n_splits = min(3, max(2, n // 5))
            tscv = TimeSeriesSplit(n_splits=n_splits)
        except Exception:
            tscv = None

        models = {}
        models["RandomForest"] = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
        models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
        if use_xgb and HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                                             subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, verbosity=0)

        for name, model in models.items():
            try:
                cv_mae = None
                if tscv is not None:
                    try:
                        scores = cross_val_score(model, X_arr, y_arr, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
                        cv_mae = float(np.mean(-scores))
                    except Exception:
                        cv_mae = None

                model.fit(X_arr, y_arr)
                preds = model.predict(X_arr)

                results[name] = {
                    "model": model,
                    "preds": np.array(preds),
                    "cv_mae": cv_mae,
                    "r2": float(r2_score(y_arr, preds)) if len(y_arr) > 0 else None,
                    "mae": float(mean_absolute_error(y_arr, preds)) if len(y_arr) > 0 else None,
                    "rmse": float(mean_squared_error(y_arr, preds, squared=False)) if len(y_arr) > 0 else None,
                    "mape": float(mean_absolute_percentage_error(y_arr, preds)) if len(y_arr) > 0 else None
                }
            except Exception as e:
                st.warning(f"Training {name} failed: {e}")
        return results

    except Exception as e:
        st.error(f"train_regressors fatal error: {e}")
        st.exception(traceback.format_exc())
        return {}

# -------------------------------
# Forecasting helper for ML
# -------------------------------
def build_future_features_from_df(df, feature_cols, start_date, months, strategy='last'):
    """
    Build a future feature DataFrame for forecasting.
    Simple strategy (safe for arbitrary data):
      - 'last': use last row values (carry-forward) for numeric features
      - 'mean': use mean of each feature
    Returns DataFrame with index 0..months-1 and 'date' column (datetime)
    """
    last_date = pd.to_datetime(df['date']).max() if 'date' in df.columns else pd.to_datetime('today')
    # align start_date to first of month if monthly
    start_dt = pd.to_datetime(start_date)
    # generate month starts
    future_dates = pd.date_range(start_dt, periods=months, freq='MS')
    # safe numeric defaults
    base_row = {}
    numeric_df = df[feature_cols].select_dtypes(include=[np.number]) if len(feature_cols)>0 else pd.DataFrame()
    mean_vals = numeric_df.mean().to_dict() if not numeric_df.empty else {}
    last_vals = numeric_df.iloc[-1].to_dict() if not numeric_df.empty else {}

    rows = []
    for i, dt in enumerate(future_dates):
        row = {}
        for f in feature_cols:
            if f in last_vals and f in mean_vals:
                if strategy == 'mean':
                    row[f] = mean_vals.get(f, 0.0)
                else:
                    # default 'last'
                    val = last_vals.get(f, mean_vals.get(f, 0.0))
                    # small jitter for variability
                    row[f] = float(val)
            else:
                row[f] = 0.0
        row['date'] = dt
        rows.append(row)
    return pd.DataFrame(rows)

def forecast_with_models(models_dict, scaler, feature_cols, df_clean, start_date, months=12, use_xgb_first=True):
    """
    Using trained models, produce forecasts for the next 'months' months.
    - models_dict: same as train_regressors() output
    - scaler: StandardScaler fit on training X (or None)
    - feature_cols: list of features used
    - df_clean: cleaned df (to derive base values)
    - start_date: datetime.date or str
    Returns:
      forecast_df: DataFrame with columns ['ds', 'model_name1', 'model_name2', ...]
      chosen_model_name: str (XGBoost if available else RandomForest)
    """
    try:
        future_X_df = build_future_features_from_df(df_clean, feature_cols, start_date, months, strategy='last')
        X_future = future_X_df[feature_cols].fillna(0.0)
        if scaler is not None:
            try:
                Xf = scaler.transform(X_future)
            except Exception:
                # fall back to values
                Xf = X_future.values
        else:
            Xf = X_future.values

        forecasts = {'ds': pd.to_datetime(future_X_df['date'])}
        for name, info in models_dict.items():
            try:
                model = info.get('model')
                if model is None:
                    forecasts[name] = [None]*len(Xf)
                else:
                    preds = model.predict(Xf)
                    forecasts[name] = [float(x) for x in preds]
            except Exception:
                forecasts[name] = [None]*len(Xf)

        # choose model to highlight: XGBoost if available and requested, else RandomForest
        chosen = None
        if use_xgb_first and 'XGBoost' in forecasts:
            chosen = 'XGBoost'
        elif 'RandomForest' in forecasts:
            chosen = 'RandomForest'
        else:
            chosen = list(models_dict.keys())[0] if models_dict else None

        forecast_df = pd.DataFrame(forecasts)
        return forecast_df, chosen
    except Exception as e:
        st.warning(f"Forecasting with models failed: {e}")
        return None, None

# -------------------------------
# 2) generate_professional_pdf_reportlab() (Option D custom)
# -------------------------------
def generate_professional_pdf_reportlab(
        df,
        company_name='Company',
        industry='Manufacturing',
        report_date=None,
        recommendations=None,
        certifications=None
    ):
    """
    Option D - custom ordered professional PDF.
    Order:
      1. Title & meta
      2. KPI section
      3. Emission Breakdown table
      4. ML performance table & ML forecast table (from st.session_state['ml_forecast'])
      5. Prophet forecast (if exists in session)
      6. ML-derived recommendations (st.session_state['ml_recommendations'])
      7. Certifications
      Footer on every page. No page breaks inserted by default.
    """

    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is not installed in this environment.")

    # local imports
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm

    buffer = io.BytesIO()
    report_date = report_date or datetime.now()

    left_margin = right_margin = 18 * mm
    top_margin = 22 * mm
    bottom_margin = 20 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin
    )

    width, height = A4
    styles = getSampleStyleSheet()

    def header_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        # Header
        canvas_obj.setFont("Helvetica-Oblique", 8)
        canvas_obj.setFillColor(colors.grey)
        canvas_obj.drawString(left_margin, height - 12 * mm, "Prepared by CodeFortune ESG Intelligence Platform")
        # Horizontal top line
        canvas_obj.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(left_margin, height - 14 * mm, width - right_margin, height - 14 * mm)
        # Footer left
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(left_margin, 12 * mm, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Page number right
        canvas_obj.drawRightString(width - right_margin, 12 * mm, f"Page {canvas_obj.getPageNumber()}")
        # bottom line
        canvas_obj.line(left_margin, 15 * mm, width - right_margin, 15 * mm)
        canvas_obj.restoreState()

    # Styles
    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=18, leading=22, alignment=0, textColor=colors.HexColor("#003366"))
    heading = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, leading=15, textColor=colors.HexColor("#002244"))
    normal = ParagraphStyle('normal', parent=styles['BodyText'], fontSize=10, leading=13)
    small = ParagraphStyle('small', parent=styles['BodyText'], fontSize=8, textColor=colors.grey)

    elems = []

    # Title
    elems.append(Paragraph(sanitize_text(company_name), title_style))
    elems.append(Paragraph("ESG Sustainability & Forecasting Report", heading))
    elems.append(Spacer(1, 6))

    # Meta
    meta = [[f"Industry: {sanitize_text(industry)}", f"Report Date: {report_date.strftime('%Y-%m-%d')}"]]
    meta_table = Table(meta, colWidths=[doc.width*0.55, doc.width*0.4])
    meta_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('ALIGN',(0,0),(-1,-1),'LEFT')]))
    elems.append(meta_table)
    elems.append(Spacer(1, 10))

    # KPI
    elems.append(Paragraph("Executive Summary", heading))
    kpis = calculate_kpis(df)
    kpi_data = [
        ['Total Emissions (kgCO2e)', f"{int(kpis['total_emissions']):,}"],
        ['Average Emission / Record', f"{kpis['avg_emissions']:.2f}"],
        ['Peak Emission', f"{int(kpis['max_emissions']):,}"],
        ['Records Analyzed', f"{kpis['records']}"]
    ]
    kpi_table = Table(kpi_data, colWidths=[doc.width*0.6, doc.width*0.35])
    kpi_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),('BOX',(0,0),(-1,-1),0.6,colors.HexColor('#bfbfbf')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f7f7f7')),('ALIGN',(1,0),(1,-1),'RIGHT')]))
    elems.append(kpi_table)
    elems.append(Spacer(1, 12))

    # Emission breakdown
    elems.append(Paragraph("Emission Breakdown", heading))
    emission_cols = [c for c in df.columns if c.endswith('_co2e')]
    breakdown_data = [['Source', 'Total (kgCO2e)']]
    if emission_cols:
        totals = df[emission_cols].sum().sort_values(ascending=False)
        for c, v in totals.items():
            breakdown_data.append([c.replace('_co2e','').replace('_',' ').title(), f"{int(v):,}"])
    else:
        breakdown_data.append(['No emission columns detected', '0'])
    breakdown_table = Table(breakdown_data, colWidths=[doc.width*0.6, doc.width*0.35])
    breakdown_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d0d0')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f7f7f7'))]))
    elems.append(breakdown_table)
    elems.append(Spacer(1, 12))

    # ML performance & forecast (if present)
    elems.append(Paragraph("Machine Learning Forecasting", heading))
    mr = st.session_state.get('model_results', {})
    if mr:
        perf = [['Model','R2','MAE','RMSE','MAPE %']]
        for name, info in mr.items():
            perf.append([
                name,
                f"{info.get('r2',0):.3f}" if info.get('r2') is not None else 'N/A',
                f"{info.get('mae',0):.2f}" if info.get('mae') is not None else 'N/A',
                f"{info.get('rmse',0):.2f}" if info.get('rmse') is not None else 'N/A',
                f"{(info.get('mape',0))*100:.2f}" if info.get('mape') is not None else 'N/A'
            ])
        perf_table = Table(perf, colWidths=[doc.width/5]*5)
        perf_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d0d0')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f5f8ff'))]))
        elems.append(perf_table)
        elems.append(Spacer(1,10))
    else:
        elems.append(Paragraph('No ML model results available.', normal))
        elems.append(Spacer(1,8))

    # ML forecast table
    ml_forecast_df = st.session_state.get('ml_forecast', None)
    if ml_forecast_df is not None and not ml_forecast_df.empty:
        # Build table: Date + each model's column; highlight chosen_model if present
        cols = ['ds'] + [c for c in ml_forecast_df.columns if c!='ds']
        ftab = [['Date'] + [c for c in cols if c!='ds']]
        for _, row in ml_forecast_df.iterrows():
            r = [pd.to_datetime(row['ds']).strftime('%Y-%m-%d')]
            for c in cols:
                if c=='ds': continue
                try:
                    val = row[c]
                    r.append(f"{float(val):.1f}" if pd.notna(val) else 'N/A')
                except Exception:
                    r.append('N/A')
            ftab.append(r)
        colw = [doc.width * 0.2] + [doc.width * 0.8 / (len(cols)-1)]*(len(cols)-1)
        mtable = Table(ftab, colWidths=colw)
        mtable.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d0d0')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f7f7f7'))]))
        elems.append(Paragraph("ML Forecast (per model)", normal))
        elems.append(mtable)
        elems.append(Spacer(1,10))

    # Prophet forecast (if exists)
    forecast_df = st.session_state.get('prophet_forecast', None)
    if forecast_df is not None and len(forecast_df)>0:
        ftab = [['Date','Predicted (kgCO2e)','Lower','Upper']]
        for _, row in forecast_df.iterrows():
            ftab.append([pd.to_datetime(row['ds']).strftime('%Y-%m-%d'),
                         f"{float(row['yhat']):.1f}",
                         f"{float(row['yhat_lower']):.1f}",
                         f"{float(row['yhat_upper']):.1f}"])
        ft = Table(ftab, colWidths=[doc.width*0.25]*4)
        ft.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d0d0')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f7f7f7'))]))
        elems.append(Paragraph("Prophet Forecast (12 months)", normal))
        elems.append(ft)
        elems.append(Spacer(1,10))

    # ML-derived recommendations (unique, not repeated)
    elems.append(Paragraph("Action Recommendations (ML-driven)", heading))
    ml_recs = st.session_state.get('ml_recommendations', [])
    if ml_recs:
        # deduplicate preserving order
        seen = set()
        unique_recs = []
        for r in ml_recs:
            if r not in seen:
                unique_recs.append(r)
                seen.add(r)
        for i, rec in enumerate(unique_recs, 1):
            elems.append(Paragraph(f"{i}. {sanitize_text(rec)}", normal))
            elems.append(Spacer(1,4))
    else:
        elems.append(Paragraph("No ML-driven recommendations available.", normal))
        elems.append(Spacer(1,6))

    # Certifications (no page break)
    if certifications:
        elems.append(Spacer(1,8))
        elems.append(Paragraph("Certifications", heading))
        cert_data = [["Certification"]]
        for c in certifications:
            cert_data.append([c])
        cert_table = Table(cert_data, colWidths=[doc.width])
        cert_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#d0d0d0')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f7f7f7'))]))
        elems.append(cert_table)
        elems.append(Spacer(1,8))

    elems.append(Spacer(1,6))
    elems.append(Paragraph("Prepared by CodeFortune ESG Intelligence Platform", small))

    # Build
    doc.build(elems, onFirstPage=header_footer, onLaterPages=header_footer)

    buffer.seek(0)
    return buffer

# ------------------ UI ------------------
st.title('Eco Sphere AI- ESG Intelligence Platform')
st.markdown('*Enterprise-grade environmental, social, and governance analytics*')

# SIDEBAR
st.sidebar.header('Configuration & Data Upload')
uploaded = st.sidebar.file_uploader('Upload Data (CSV / Excel)', type=['csv','xlsx'], help='Upload your CSV or XLSX dataset exported from your system.')
use_demo = st.sidebar.checkbox('Use Demo Dataset', value=True)
demo_months = st.sidebar.number_input('Demo months', min_value=12, max_value=240, value=120, step=12)
company_input = st.sidebar.text_input('Company name (report)', value='Company A')
industry_input = st.sidebar.text_input('Industry (report)', value='Manufacturing')
report_date_input = st.sidebar.date_input('Report date', value=date.today())
company_choice = st.sidebar.selectbox('Demo Company', ['Company A','Company B','Custom'])
industry_choice = st.sidebar.selectbox('Demo Industry', [k.title() for k in INDUSTRY_BENCHMARKS.keys()])

# Prepare demo datasets (cached)
demo_a = generate_demo_dataset('Company A', start='2016-01-01', months=int(demo_months), seed=42, industry=industry_choice.lower())
demo_b = generate_demo_dataset('Company B', start='2018-01-01', months=int(max(12, demo_months//1.3)), seed=24, industry=industry_choice.lower())

# Load uploaded file robustly
if uploaded is not None:
    df_uploaded = load_uploaded_file(uploaded, default_industry=industry_input, default_company=company_input)
    if df_uploaded is not None:
        if industry_input:
            df_uploaded['industry'] = industry_input.lower()
        if company_input:
            df_uploaded['company'] = company_input
        cols = df_uploaded.columns.tolist()
        for c in ['industry','company']:
            if c in cols:
                cols.remove(c)
                cols.insert(0,c)
        df_uploaded = df_uploaded[cols]
        st.session_state['uploaded_df'] = df_uploaded
    else:
        st.session_state['uploaded_df'] = None
else:
    if 'uploaded_df' not in st.session_state:
        st.session_state['uploaded_df'] = None

# FIXED RAW DATA LOADER (no DataFrame boolean)
raw_df = st.session_state.get('uploaded_df')
if isinstance(raw_df, pd.DataFrame) and raw_df.empty:
    raw_df = None
if raw_df is None:
    if use_demo:
        raw_df = demo_a if company_choice == 'Company A' else demo_b
    else:
        raw_df = None

# UI Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Data Overview','Data Cleaning','Analytics','ML Forecasting','Compliance','Export/Reports'])

with tab1:
    st.header('Data Preview & Overview')
    if raw_df is not None:
        try:
            c1,c2,c3 = st.columns(3)
            with c1:
                st.metric('Records', len(raw_df))
            with c2:
                st.metric('Columns', len(raw_df.columns))
            with c3:
                date_cols = [c for c in raw_df.columns if 'date' in c]
                if date_cols:
                    try:
                        dr = pd.to_datetime(raw_df[date_cols[0]], errors='coerce')
                        dr_nonnull = dr.dropna()
                        if len(dr_nonnull) > 0:
                            st.metric('Date Range', f"{dr_nonnull.min().strftime('%Y-%m-%d')} → {dr_nonnull.max().strftime('%Y-%m-%d')}")
                        else:
                            st.metric('Date Range', 'Unknown')
                    except Exception:
                        st.metric('Date Range', f"{len(raw_df)} records")
                else:
                    st.metric('Date Range', f"{len(raw_df)} records")

            st.subheader('Data Sample')
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.subheader('Data Quality Check')
            missing = raw_df.isnull().sum()
            if missing.sum() > 0:
                st.warning(f'Missing values detected: {dict(missing[missing > 0])}')
            else:
                st.success('No missing values detected')
        except Exception as e:
            st.error(f'Error while rendering preview: {e}')
            st.exception(traceback.format_exc())
    else:
        st.error('No data available. Upload a file or enable demo dataset.')

with tab2:
    st.header('Data Cleaning')
    if raw_df is not None:
        try:
            df_clean, numeric_cols = clean_and_engineer(raw_df)
            if df_clean is None:
                st.error('Cleaning failed; ensure your file has usable rows and columns.')
            else:
                st.session_state['cleaned_df'] = df_clean
                st.success('Data cleaned and features engineered successfully')

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric('Total Rows', len(df_clean))
                with c2:
                    st.metric('Numeric Features', len(numeric_cols))
                with c3:
                    total_em = int(df_clean['emission_kgco2e'].sum()) if 'emission_kgco2e' in df_clean.columns else 0
                    st.metric('Emission Total', f"{total_em:,} kg")
                with c4:
                    avg_em = df_clean['emission_kgco2e'].mean() if 'emission_kgco2e' in df_clean.columns else 0
                    st.metric('Avg Emission', f"{avg_em:.2f} kg")

                st.subheader('Summary Statistics (numeric)')
                if numeric_cols:
                    st.dataframe(df_clean[numeric_cols].describe().T, use_container_width=True)
                else:
                    st.info('No numeric columns to display')

                df_export = df_clean.copy()
                for col in ['company','industry']:
                    if col in df_export.columns:
                        df_export.drop(columns=[col], inplace=True)
                csv_buf = io.StringIO()
                df_export.to_csv(csv_buf, index=False)
                st.download_button('Download Cleaned Data (CSV)', csv_buf.getvalue(), file_name=f'cleaned_{company_input}.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Error during cleaning: {e}')
            st.exception(traceback.format_exc())
    else:
        st.error('No data to clean')

with tab3:
    st.header('Analytics Dashboard & KPIs')
    if st.session_state['cleaned_df'] is not None:
        dfc = st.session_state['cleaned_df']
        kpis = calculate_kpis(dfc)

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric('Total Emissions', f"{int(kpis['total_emissions']):,} kg")
        with c2:
            st.metric('Average Emission', f"{kpis['avg_emissions']:.2f} kg")
        with c3:
            st.metric('Peak Emission', f"{int(kpis['max_emissions']):,} kg")
        with c4:
            label = '📈 Worsening' if kpis['trend']=='worsening' else '📉 Improving'
            st.metric('Trend', label)

        emission_cols = [c for c in dfc.columns if c.endswith('_co2e')]
        if emission_cols:
            st.subheader('Emission Composition')
            breakdown = dfc[emission_cols].sum().sort_values(ascending=False)
            labels = [c.replace('_co2e','').replace('_',' ').title() for c in breakdown.index]
            if HAS_PLOTLY:
                fig = go.Figure(data=[go.Pie(labels=labels, values=breakdown.values, hole=0.3)])
                fig.update_layout(title='Emission Sources')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.pie(breakdown.values, labels=labels, autopct='%1.1f%%')
                ax.set_title('Emission Sources')
                st.pyplot(fig)

        st.subheader('Emissions Over Time')
        if 'emission_kgco2e' in dfc.columns:
            if HAS_PLOTLY:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=dfc['date'], y=dfc['emission_kgco2e'], mode='lines+markers', name='Emissions'))
                fig_ts.update_layout(title='Emissions Trend', xaxis_title='Date', yaxis_title='kgCO2e')
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                fig2, ax2 = plt.subplots()
                ax2.plot(dfc['date'], dfc['emission_kgco2e'], marker='o', linewidth=1)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('kgCO2e')
                plt.xticks(rotation=30)
                st.pyplot(fig2)
    else:
        st.error('Please clean data first in Tab 2')

# -------------------------------
# TAB 4 (ML Forecasting & Recommendations)
# -------------------------------
with tab4:
    st.header('ML Forecasting & Recommendations')
    if st.session_state.get('cleaned_df') is None:
        st.error('Please clean data first in Tab 2')
    else:
        dfc = st.session_state['cleaned_df'].copy()
        if 'emission_kgco2e' not in dfc.columns:
            st.info("No 'emission_kgco2e' column found for ML training.")
        else:
            df_ml = dfc.copy()
            for c in ['company','industry']:
                if c in df_ml.columns:
                    df_ml.drop(columns=[c], inplace=True)

            feature_cols = [c for c in df_ml.select_dtypes(include=[np.number]).columns.tolist() if c not in ['emission_kgco2e']]
            if not feature_cols:
                st.info('No numeric features available for ML training.')
            else:
                X = df_ml[feature_cols].fillna(0)
                y = df_ml['emission_kgco2e']

                st.subheader('Forecast settings')
                last_date = pd.to_datetime(dfc['date']).max() if 'date' in dfc.columns else pd.to_datetime('today')
                default_start = last_date + pd.offsets.MonthBegin(1)
                forecast_start = st.date_input('Forecast start date (first month)', value=default_start)
                forecast_months = st.number_input('Forecast months', min_value=1, max_value=60, value=12, step=1)
                carbon_price_per_tonne = st.number_input('Carbon price (USD per tonne) for savings calc', value=50.0, step=1.0)

                if st.button('Train Models (XGBoost/RF/GB)', key='train_models'):
                    with st.spinner('Training models...'):
                        try:
                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(X)

                            results = train_regressors(Xs, y)
                            st.session_state['model_results'] = results
                            st.session_state['scaler'] = scaler
                            st.session_state['feature_cols'] = feature_cols
                            st.session_state['dates'] = pd.to_datetime(df_ml['date']) if 'date' in df_ml.columns else np.arange(len(df_ml))
                            st.session_state['y'] = y.values

                            # Forecast with models
                            use_xgb_first = True
                            fc_df, chosen = forecast_with_models(results, scaler, feature_cols, dfc, forecast_start, months=forecast_months, use_xgb_first=use_xgb_first)
                            if fc_df is not None:
                                st.session_state['ml_forecast'] = fc_df
                            else:
                                st.session_state['ml_forecast'] = None

                            # Build ML driven recommendations based on forecast and baseline
                            baseline_total = float(dfc['emission_kgco2e'].sum())
                            baseline_avg = float(dfc['emission_kgco2e'].mean())
                            ml_recs = []
                            if st.session_state.get('ml_forecast') is not None:
                                # chosen model forecast average vs baseline
                                chosen_model = chosen
                                try:
                                    fc_chosen_avg = float(np.mean(st.session_state['ml_forecast'][chosen_model].astype(float)))
                                except Exception:
                                    fc_chosen_avg = None

                                if fc_chosen_avg is not None:
                                    if fc_chosen_avg > baseline_avg * 1.05:
                                        ml_recs.append(f"ML models predict a rising trend (avg forecast {fc_chosen_avg:.1f} kg > baseline avg {baseline_avg:.1f} kg). Investigate peak drivers (energy use, production spikes).")
                                    else:
                                        reduction_pct = max(5, min(30, int(100*(1 - fc_chosen_avg / baseline_avg)))) if baseline_avg>0 else 10
                                        saved_emission = baseline_total * (reduction_pct/100)
                                        saved_tonnes = saved_emission/1000.0
                                        monetary_saving = saved_tonnes * carbon_price_per_tonne
                                        ml_recs.append(f"Target a {reduction_pct}% reduction in emissions vs baseline — estimated saving {int(saved_emission):,} kg CO2e (~{saved_tonnes:.2f} t) ≈ ${monetary_saving:,.2f} at ${carbon_price_per_tonne}/t.")
                                        ml_recs.append("Focus on high-impact interventions: energy efficiency, fuel optimization, process scheduling.")
                                else:
                                    ml_recs.append("ML forecast computed but could not derive actionable average; review input features.")
                            else:
                                ml_recs.append("No ML forecast available to create ML-driven recommendations.")

                            st.session_state['ml_recommendations'] = ml_recs
                            st.success('Models trained and ML forecast generated successfully')
                        except Exception as e:
                            st.error(f'Model training failed: {e}')
                            st.exception(traceback.format_exc())

                if st.session_state.get('model_results'):
                    st.subheader('Model Performance')
                    res = st.session_state['model_results']
                    summary = []
                    for k,v in res.items():
                        summary.append({'Model': k,
                                        'R2': f"{v['r2']:.4f}" if v.get('r2') is not None else 'N/A',
                                        'MAE': f"{v['mae']:.2f}" if v.get('mae') is not None else 'N/A',
                                        'RMSE': f"{v['rmse']:.2f}" if v.get('rmse') is not None else 'N/A',
                                        'MAPE': f"{v['mape']*100:.2f}%" if v.get('mape') is not None else 'N/A'})
                    st.dataframe(pd.DataFrame(summary), use_container_width=True)

                    # show predictions training set
                    pred_df = pd.DataFrame()
                    pred_df['date'] = pd.to_datetime(df_ml['date']) if 'date' in df_ml.columns else range(len(df_ml))
                    scaler = st.session_state.get('scaler', None)
                    for name, info in res.items():
                        model = info['model']
                        try:
                            Xs = scaler.transform(X) if scaler is not None else X
                        except Exception:
                            Xs = X
                        try:
                            preds = model.predict(Xs)
                        except Exception:
                            preds = np.zeros(len(X))
                        pred_df[f'pred_{name}'] = preds
                    pred_df['actual_emission'] = y.values

                    st.subheader('Predictions (training set sample)')
                    st.dataframe(pred_df.head(50), use_container_width=True)
                    csv_buf = io.StringIO()
                    pred_df.to_csv(csv_buf, index=False)
                    st.download_button('Download Predictions CSV', csv_buf.getvalue(), file_name=f'{company_input}_predictions.csv', mime='text/csv')

                    # show ml_forecast table (if exists)
                    if st.session_state.get('ml_forecast') is not None:
                        st.subheader('ML Forecast (per model)')
                        st.dataframe(st.session_state['ml_forecast'], use_container_width=True)
                        csv_b = io.StringIO()
                        st.session_state['ml_forecast'].to_csv(csv_b, index=False)
                        st.download_button('Download ML Forecast CSV', csv_b.getvalue(), file_name=f'{company_input}_ml_forecast.csv', mime='text/csv')

                    # show ML-driven recommendations and allow add to report
                    st.subheader('ML-driven Recommendations')
                    ml_recs = st.session_state.get('ml_recommendations', [])
                    if ml_recs:
                        for rec in ml_recs:
                            st.write(f"- {rec}")
                        if st.button('Add ML recommendations to report'):
                            if 'recommendations' not in st.session_state:
                                st.session_state['recommendations'] = []
                            for rec in ml_recs:
                                if rec not in st.session_state['recommendations']:
                                    st.session_state['recommendations'].append(rec)
                            st.success('ML recommendations added to report')
                    else:
                        st.info('No ML recommendations available yet.')

# -------------------------------
# TAB 5 (Compliance & Benchmarking)
# -------------------------------
with tab5:
    st.header('Compliance & Industry Benchmarking')

    if st.session_state.get('cleaned_df') is None:
        st.error('No data available')
    else:
        dfc = st.session_state['cleaned_df']
        c1,c2 = st.columns(2)
        with c1:
            st.subheader('Certifications (sample status)')
            cert_status = {k:'In Progress' for k in ESG_CERTIFICATIONS.keys()}
            cert_status['Carbon Trust'] = 'Certified'
            for cert, status in cert_status.items():
                icon = '✅' if status=='Certified' else '⏳'
                st.write(f'{icon} {cert}: {status}')
        with c2:
            st.subheader('Industry Benchmarking')
            avg_em = dfc['emission_kgco2e'].mean() if 'emission_kgco2e' in dfc.columns else 0
            industry_benchmark = INDUSTRY_BENCHMARKS.get(industry_choice.lower(), 5.0)
            try:
                performance = (1 - (avg_em / (industry_benchmark*1000))) * 100
                st.metric('Your Avg Emission', f'{avg_em:.2f} kg', delta=f'{performance:.1f}% vs industry avg')
            except Exception:
                st.metric('Your Avg Emission', f'{avg_em:.2f} kg')
            bench_df = pd.DataFrame({'Metric':['Your Company','Industry Avg'],'Emissions':[avg_em, industry_benchmark*1000]})
            if HAS_PLOTLY:
                figb = px.bar(bench_df, x='Metric', y='Emissions', title='Benchmarking')
                st.plotly_chart(figb, use_container_width=True)
            else:
                figb, axb = plt.subplots()
                axb.bar(bench_df['Metric'], bench_df['Emissions'])
                st.pyplot(figb)
        st.subheader('Regulatory Compliance Checklist')
        compliance_items = [('GHG Protocol Corporate Standard', True), ('TCFD Climate Risk Disclosure', True), ('SEC Climate Disclosure Rule', False)]
        st.dataframe(pd.DataFrame(compliance_items, columns=['Requirement','Compliant']), use_container_width=True)
# -------------------------------
# 5) TAB 6 replacement (Export & Professional Reports)
# -------------------------------
with tab6:
    st.header('Export & Reports')

    if st.session_state.get('cleaned_df') is None:
        st.error('No data available for export')
    else:
        dfc = st.session_state['cleaned_df']

        # ------------------------------------------------
        # FIX: Detect whether user is using demo or upload
        # ------------------------------------------------
        is_demo = (uploaded is None and use_demo)

        # ------------------------------------------------
        # FIX: Determine company/industry correctly
        # ------------------------------------------------
        if is_demo:
            # Use DEMO company/industry
            company_for_pdf = company_choice      # Company A/B/Custom
            industry_for_pdf = industry_choice    # Industry selected in dropdown
        else:
            # Using uploaded data → use sidebar OR dataset metadata
            company_for_pdf = (
                company_input
                if company_input else
                (dfc['company'].iloc[0] if 'company' in dfc.columns else 'Company')
            )
            industry_for_pdf = (
                industry_input
                if industry_input else
                (dfc['industry'].iloc[0] if 'industry' in dfc.columns else industry_choice)
            )

        # ------------------------------------------------
        # Preview data BEFORE exporting
        # ------------------------------------------------
        c0, c1 = st.columns([1,1])
        with c0:
            st.write("**Company used in PDF:**", company_for_pdf)
        with c1:
            st.write("**Industry used in PDF:**", industry_for_pdf)

        st.subheader('Generate PDF')

        recs = st.session_state.get('recommendations', [
            'Implement LED lighting retrofit to reduce electricity consumption',
            'Schedule preventive maintenance to avoid fuel spikes',
            'Investigate waste reduction programs on high-waste months'
        ])

        certifications_list = list(ESG_CERTIFICATIONS.keys()) if ESG_CERTIFICATIONS else None

        # -----------------------------
        # PDF Generation Button
        # -----------------------------
        if st.button('Generate ESG PDF Report'):
            if not HAS_REPORTLAB:
                st.error('ReportLab is not installed; cannot generate PDF here.')
            else:
                with st.spinner('Creating PDF report...'):
                    try:
                        buf = generate_professional_pdf_reportlab(
                            dfc,
                            company_name=company_for_pdf,
                            industry=industry_for_pdf,
                            report_date=report_date_input,
                            recommendations=recs,
                            certifications=certifications_list
                        )

                        st.download_button(
                            'Download PDF Report',
                            data=buf.getvalue(),
                            file_name=f'{company_for_pdf}_ESG_Report.pdf',
                            mime='application/pdf'
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {e}")
                        st.exception(traceback.format_exc())

        st.markdown("---")
        st.subheader("Download Cleaned Data")

        df_export = dfc.copy()
        for col in ['company', 'industry']:
            if col in df_export.columns:
                df_export.drop(columns=[col], inplace=True)

        csv_buf = io.StringIO()
        df_export.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Cleaned CSV",
            csv_buf.getvalue(),
            file_name=f"{company_for_pdf}_cleaned_data.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Export Summary")
        export_summary = {
            'Total Records': len(dfc),
            'Total Emissions (kg CO2e)': f"{int(dfc['emission_kgco2e'].sum()):,}" if 'emission_kgco2e' in dfc.columns else '0',
            'Average Emissions': f"{dfc['emission_kgco2e'].mean():.2f}" if 'emission_kgco2e' in dfc.columns else '0',
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.json(export_summary)
