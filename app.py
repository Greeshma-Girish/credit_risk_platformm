import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.data.loader import load_data
from src.ml.predict import load_model, predict, explain_prediction
from src.talk_to_data.nl_to_sql import ask_data
from src.data.preprocessor import preprocess_data
from explainability.explainer import get_human_explanation, get_feature_importance

# 1. PAGE CONFIGURATION & GLOBAL DESIGN SYSTEM
st.set_page_config(
    page_title="AI Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Modern CSS (Vibrant Colors, Glassmorphism, Custom Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Glassmorphic Metric Cards */
    .metric-card-wrapper {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 25px;
    }
    .vibrant-card {
        background: #1E293B !important;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155 !important;
        transition: all 0.25s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    .vibrant-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
    }
    .vibrant-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
    }
    .card-purple::before { background: linear-gradient(90deg, #8B5CF6, #C084FC); }
    .card-emerald::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .card-blue::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .card-rose::before { background: linear-gradient(90deg, #F43F5E, #FB7185); }
    
    .card-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .card-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F8FAFC !important;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-pill {
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .badge-low-risk {
        background: #064E3B !important;
        color: #6EE7B7 !important;
        border: 1px solid #059669 !important;
    }
    .badge-medium-risk {
        background: #78350F !important;
        color: #FDE047 !important;
        border: 1px solid #D97706 !important;
    }
    .badge-high-risk {
        background: #7F1D1D !important;
        color: #FCA5A5 !important;
        border: 1px solid #DC2626 !important;
    }

    /* Recommendation Banner */
    .rec-box {
        padding: 16px 20px;
        border-radius: 12px;
        color: white !important;
        font-weight: 800;
        font-size: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .rec-approve { background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important; }
    .rec-review { background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%) !important; }
    .rec-decline { background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%) !important; }

    /* Policy Rules Card */
    .policy-rule-card {
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .rule-pass { 
        border-left: 6px solid #10B981 !important; 
        background: #064E3B !important; 
        color: #ECFDF5 !important; 
    }
    .rule-pass strong, .rule-pass b, .rule-pass span, .rule-pass div {
        color: #ECFDF5 !important;
    }
    .rule-fail { 
        border-left: 6px solid #EF4444 !important; 
        background: #7F1D1D !important; 
        color: #FEF2F2 !important; 
    }
    .rule-fail strong, .rule-fail b, .rule-fail span, .rule-fail div {
        color: #FEF2F2 !important;
    }

    /* Insight Cards */
    .insight-card {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #8B5CF6 !important;
        border: 1px solid #334155;
    }
    .insight-card strong {
        color: #A78BFA !important;
    }

</style>
""", unsafe_allow_html=True)

# 2. DATA & MODEL LOADING (CACHED)
@st.cache_data
def get_full_data():
    return load_data(is_train=True)

@st.cache_data
def get_sample_data():
    df = get_full_data()
    return df.sample(n=2000, random_state=42)

@st.cache_resource
def get_ml_model():
    return load_model()

# 3. NAVIGATION & HEADER
# Hero Header
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🏦 AI Credit Risk Intelligence Platform</div>
    <div class='hero-subtitle'>Enterprise Decisioning Engine • Explainable AI (SHAP) • Talk-to-Data Analytics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Module Navigation")
page = st.sidebar.radio(
    "Select Feature",
    [
        "📊 Exploratory Data Analysis",
        "🎯 Risk Prediction",
        "🔍 Explainability (XAI)",
        "📜 Underwriting Policy & Rules",
        "💬 Talk-to-Data Chatbot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: #1E293B; color: #E2E8F0; padding: 14px; border-radius: 10px; font-size: 0.85rem;'>
    <strong style='color: #818CF8;'>⚙️ Model Status:</strong> Active<br>
    <strong style='color: #818CF8;'>Model Engine:</strong> LightGBM Classifier<br>
    <strong style='color: #818CF8;'>Optimization:</strong> Cost-sensitive Weighted Loss (`scale_pos_weight`)
</div>
""", unsafe_allow_html=True)

# PAGE 1: Exploratory Data Analysis (EDA)
if page == "📊 Exploratory Data Analysis":
    st.markdown("## 📊 Exploratory Data Analytics (EDA)")
    st.write("Deep-dive exploration into portfolio credit risk, applicant demographics, and default distribution.")
    
    try:
        df = get_sample_data()
        
        # Metric Cards Header
        st.markdown(f"""
        <div class='metric-card-wrapper'>
            <div class='vibrant-card card-purple'>
                <div class='card-label'>Sample Size</div>
                <div class='card-value'>2,000</div>
            </div>
            <div class='vibrant-card card-rose'>
                <div class='card-label'>Portfolio Default Rate</div>
                <div class='card-value'>{df['TARGET'].mean() * 100:.2f}%</div>
            </div>
            <div class='vibrant-card card-blue'>
                <div class='card-label'>Avg Loan Credit</div>
                <div class='card-value'>${df['AMT_CREDIT'].mean():,.0f}</div>
            </div>
            <div class='vibrant-card card-emerald'>
                <div class='card-label'>Avg Income Total</div>
                <div class='card-value'>${df['AMT_INCOME_TOTAL'].mean():,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Target Imbalance", 
            "💰 Financial Profiles", 
            "👥 Applicant Demographics", 
            "🧹 Data Quality Audit", 
            "💡 Executive Insights"
        ])
        
        with tab1:
            st.markdown("### Portfolio Outcome Breakdown")
            c_a, c_b = st.columns([1, 2])
            with c_a:
                counts = df['TARGET'].value_counts()
                st.markdown(f"""
                - **Non-Default (0):** `{counts.get(0, 0):,}` ({counts.get(0, 0)/len(df):.1%})
                - **Default (1):** `{counts.get(1, 0):,}` ({counts.get(1, 0)/len(df):.1%})
                """)
                st.info("ℹ️ **Class Imbalance Note:** The default rate is ~8.07%. Standard classifiers require cost-sensitive reweighting to prevent high false negative rates.")
            with c_b:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                colors = ["#10B981", "#EF4444"]
                counts.plot(kind="bar", color=colors, ax=ax)
                ax.set_xticklabels(["Repaid (0)", "Default (1)"], rotation=0, fontweight="bold")
                ax.set_ylabel("Applicant Count")
                ax.set_title("Loan Default Outcome Distribution", fontweight="bold", color="#1E293B")
                st.pyplot(fig)

        with tab2:
            st.markdown("### Income & Requested Credit Distribution")
            f1, f2 = st.columns(2)
            with f1:
                p99 = df['AMT_INCOME_TOTAL'].quantile(0.99)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(df[df['AMT_INCOME_TOTAL'] <= p99]['AMT_INCOME_TOTAL'], bins=35, color="#6366F1", edgecolor="#312E81")
                ax.set_xlabel("Income ($)")
                ax.set_ylabel("Count")
                ax.set_title("Applicant Income (up to 99th Percentile)", fontweight="bold")
                st.pyplot(fig)
            with f2:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(df['AMT_CREDIT'], bins=35, color="#8B5CF6", edgecolor="#4C1D95")
                ax.set_xlabel("Requested Credit Amount ($)")
                ax.set_ylabel("Count")
                ax.set_title("Credit Amount Distribution", fontweight="bold")
                st.pyplot(fig)

        with tab3:
            st.markdown("### Risk Correlations across Demographics")
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Default Rate by Education Level:**")
                edu_df = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 3.8))
                edu_df.sort_values().plot(kind="barh", color="#F59E0B", ax=ax)
                ax.set_xlabel("Default Rate (%)")
                st.pyplot(fig)
            with d2:
                st.write("**Default Rate by Housing Type:**")
                housing_df = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean() * 100
                fig, ax = plt.subplots(figsize=(6, 3.8))
                housing_df.sort_values().plot(kind="barh", color="#06B6D4", ax=ax)
                ax.set_xlabel("Default Rate (%)")
                st.pyplot(fig)

        with tab4:
            st.markdown("### Missing Feature Distribution")
            missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            top_missing = missing_pct[missing_pct > 0].head(12)
            
            fig, ax = plt.subplots(figsize=(8, 3.8))
            top_missing.plot(kind="barh", color="#EC4899", ax=ax)
            ax.set_xlabel("Missing Rate (%)")
            ax.set_title("Top 12 Features with Missing Data", fontweight="bold")
            st.pyplot(fig)

        with tab5:
            st.markdown("### Key Business Insights")
            insights = [
                "**1. Class Imbalance Awareness**: ~8% default rate requires cost-weighted loss (`scale_pos_weight`) during ML training.",
                "**2. Debt-to-Income Leverage**: Loan requests > 5x annual applicant income carry a 3x higher default risk.",
                "**3. Educational Attainment**: Higher education correlates strongly with lower default rates compared to lower secondary education.",
                "**4. Residential Stability**: Homeowners exhibit lower default rates than applicants living in rented apartments or with parents.",
                "**5. External Scores**: `EXT_SOURCE_2` and `EXT_SOURCE_3` are the top negative predictors of credit default."
            ]
            for ins in insights:
                st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error in EDA module: {e}")

# PAGE 2: Risk Prediction
elif page == "🎯 Risk Prediction":
    st.markdown("## 🎯 Real-Time Credit Risk Decisioning")
    st.write("Compute instantaneous default probabilities and underwriting risk tiers for applicants.")
    
    try:
        df = get_sample_data()
        model = get_ml_model()
        
        mode = st.radio("Choose Mode:", ["Lookup Existing Applicant", "Custom Applicant Simulator"], horizontal=True)
        
        if mode == "Lookup Existing Applicant":
            applicant_id = st.selectbox("Select Applicant ID (SK_ID_CURR):", df['SK_ID_CURR'].tolist())
            applicant_row = df[df['SK_ID_CURR'] == applicant_id]
            
            st.markdown("#### 👤 Applicant Financial Summary")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Annual Income", f"${applicant_row['AMT_INCOME_TOTAL'].values[0]:,.0f}")
            p2.metric("Requested Credit", f"${applicant_row['AMT_CREDIT'].values[0]:,.0f}")
            p3.metric("Annual Annuity", f"${applicant_row['AMT_ANNUITY'].values[0]:,.0f}")
            p4.metric("Education", f"{applicant_row['NAME_EDUCATION_TYPE'].values[0]}")
            
            X, _ = preprocess_data(applicant_row, is_train=False)
            
        else:
            st.markdown("#### 🎛️ Custom Applicant Profile Builder")
            c1, c2, c3 = st.columns(3)
            with c1:
                income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=120000, step=5000)
                credit = st.number_input("Requested Credit ($)", min_value=10000, max_value=5000000, value=400000, step=10000)
            with c2:
                annuity = st.number_input("Annual Annuity ($)", min_value=1000, max_value=500000, value=20000, step=1000)
                age_years = st.slider("Applicant Age", 18, 70, 32)
            with c3:
                education = st.selectbox("Education Level", df['NAME_EDUCATION_TYPE'].unique())
                housing = st.selectbox("Housing Type", df['NAME_HOUSING_TYPE'].unique())

            applicant_row = df.iloc[[0]].copy()
            applicant_row['AMT_INCOME_TOTAL'] = income
            applicant_row['AMT_CREDIT'] = credit
            applicant_row['AMT_ANNUITY'] = annuity
            applicant_row['DAYS_BIRTH'] = -int(age_years * 365)
            applicant_row['NAME_EDUCATION_TYPE'] = education
            applicant_row['NAME_HOUSING_TYPE'] = housing
            
            X, _ = preprocess_data(applicant_row, is_train=False)

        if st.button("🚀 Evaluate Underwriting Decision", type="primary"):
            with st.spinner("Processing LightGBM inference engine..."):
                probs, bands = predict(model, X)
                score = probs[0]
                band = bands[0]
                
                st.markdown("---")
                st.markdown("### 📊 Underwriting Assessment Result")
                
                r1, r2, r3 = st.columns(3)
                
                with r1:
                    st.metric("Default Probability Score", f"{score:.1%}")
                
                with r2:
                    if band == "Low":
                        b_cls = "badge-low-risk"
                        b_icn = "🟢"
                    elif band == "Medium":
                        b_cls = "badge-medium-risk"
                        b_icn = "🟡"
                    else:
                        b_cls = "badge-high-risk"
                        b_icn = "🔴"
                    st.markdown(f"**Risk Tier:**<br><span class='badge-pill {b_cls}'>{b_icn} {band.upper()} RISK</span>", unsafe_allow_html=True)
                
                with r3:
                    if band == "Low":
                        st.markdown("<div class='rec-box rec-approve'>✅ LOAN APPROVED</div>", unsafe_allow_html=True)
                    elif band == "Medium":
                        st.markdown("<div class='rec-box rec-review'>⚠️ MANUAL UNDERWRITING</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='rec-box rec-decline'>❌ LOAN DECLINED</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error executing risk prediction: {e}")

# PAGE 3: Explainability (XAI)
elif page == "🔍 Explainability (XAI)":
    st.markdown("## 🔍 Explainable AI (SHAP Insights)")
    st.write("Transparent model decision auditing powered by SHAP (SHapley Additive exPlanations).")
    
    try:
        df = get_sample_data()
        model = get_ml_model()
        
        applicant_id = st.selectbox("Select Applicant ID to Audit:", df['SK_ID_CURR'].tolist())
        applicant_data = df[df['SK_ID_CURR'] == applicant_id]
        X, _ = preprocess_data(applicant_data, is_train=False)
        
        explainer, shap_values = explain_prediction(model, X)
        
        st.markdown("### 1. Human-Readable Risk Factors")
        explanations = get_human_explanation(X, shap_values, top_n=6)
        
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("#### 🔴 Risk Accelerators (Pushed Score Higher)")
            has_pos = False
            for exp in explanations:
                if exp['direction'] == 'increased':
                    st.markdown(f"• **{exp['feature']}**: Increased risk by `+{exp['impact']:.4f}`")
                    has_pos = True
            if not has_pos:
                st.write("No major factors increased default risk.")

        with e2:
            st.markdown("#### 🟢 Risk Mitigators (Pushed Score Lower)")
            has_neg = False
            for exp in explanations:
                if exp['direction'] == 'decreased':
                    st.markdown(f"• **{exp['feature']}**: Mitigated risk by `-{exp['impact']:.4f}`")
                    has_neg = True
            if not has_neg:
                st.write("No major mitigating factors found.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 2. SHAP Feature Contribution Waterfall / Bar Plot")
        
        fig, ax = plt.subplots(figsize=(9, 4.2))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in explainability module: {e}")

# PAGE 4: Policy Rules Engine
elif page == "📜 Underwriting Policy & Rules":
    st.markdown("## 📜 Policy Rules & Underwriting Compliance Engine")
    st.write("Combine machine learning risk scores with hard business rules and policy constraints.")
    
    df = get_sample_data()
    applicant_id = st.selectbox("Select Applicant ID for Policy Audit:", df['SK_ID_CURR'].tolist())
    app_data = df[df['SK_ID_CURR'] == applicant_id].iloc[0]
    
    st.markdown("### ⚙️ Policy Parameter Controls")
    u1, u2 = st.columns(2)
    with u1:
        max_ratio = st.slider("Max Credit-to-Income Multiple Ceiling", 1.0, 10.0, 5.0, 0.5)
        max_ann = st.slider("Max Annuity Burden Ratio (%)", 10.0, 60.0, 40.0, 5.0)
    with u2:
        min_inc = st.number_input("Minimum Income Floor ($)", value=25000, step=5000)
        min_age = st.slider("Minimum Age Requirement", 18, 25, 21)

    st.markdown("---")
    st.markdown("### 📋 Automated Compliance Audit Result")
    
    inc = app_data['AMT_INCOME_TOTAL']
    cred = app_data['AMT_CREDIT']
    ann = app_data['AMT_ANNUITY']
    age = -app_data['DAYS_BIRTH'] / 365
    
    ratio = cred / inc if inc > 0 else 0
    ann_ratio = (ann / inc) * 100 if inc > 0 else 0

    p1 = ratio <= max_ratio
    p2 = ann_ratio <= max_ann
    p3 = inc >= min_inc
    p4 = age >= min_age

    st.markdown(f"""
    <div class='policy-rule-card {"rule-pass" if p1 else "rule-fail"}'>
        <strong>Rule 1: Credit-to-Income Multiple ({ratio:.2f}x vs Max {max_ratio}x)</strong><br>
        Status: {"✅ PASS" if p1 else "❌ FAIL - Excessive Debt Multiple"}
    </div>
    <div class='policy-rule-card {"rule-pass" if p2 else "rule-fail"}'>
        <strong>Rule 2: Payment Burden Ratio ({ann_ratio:.1f}% vs Max {max_ann}%)</strong><br>
        Status: {"✅ PASS" if p2 else "❌ FAIL - Payment Burden Exceeds Limit"}
    </div>
    <div class='policy-rule-card {"rule-pass" if p3 else "rule-fail"}'>
        <strong>Rule 3: Minimum Income Requirement (${inc:,.0f} vs Min ${min_inc:,.0f})</strong><br>
        Status: {"✅ PASS" if p3 else "❌ FAIL - Income Below Policy Floor"}
    </div>
    <div class='policy-rule-card {"rule-pass" if p4 else "rule-fail"}'>
        <strong>Rule 4: Minimum Age Requirement ({age:.0f} yrs vs Min {min_age} yrs)</strong><br>
        Status: {"✅ PASS" if p4 else "❌ FAIL - Age Eligibility Failed"}
    </div>
    """, unsafe_allow_html=True)

    passed = sum([p1, p2, p3, p4])
    if passed == 4:
        st.success("🎉 **VERDICT: FULL COMPLIANCE PASS** — Meets all hard policy constraints.")
    elif passed >= 2:
        st.warning("⚠️ **VERDICT: CONDITIONAL REVIEW** — Soft policy flags triggered.")
    else:
        st.error("🚨 **VERDICT: HARD REJECT** — Failed multiple core underwriting rules.")

# PAGE 5: Talk-to-Data Chatbot
elif page == "💬 Talk-to-Data Chatbot":
    st.markdown("## 💬 Talk-to-Data SQL Analytics")
    st.write("Convert natural language business queries into executable SQL using Gemini AI.")
    
    st.markdown("#### 💡 Quick Prompt Suggestions")
    q_cols = st.columns(3)
    
    prompt = st.text_input("Enter natural language question:", placeholder="e.g. What is the average credit amount for applicants who defaulted?")
    
    if q_cols[0].button("📊 Average Credit by Default"):
        prompt = "What is the average credit amount for applicants who defaulted?"
    if q_cols[1].button("💰 Income by Education"):
        prompt = "Compare the average income by education level."
    if q_cols[2].button("👥 Default Count Breakdown"):
        prompt = "How many applicants defaulted vs non-defaulted?"

    if prompt:
        with st.spinner("🤖 Translating query with Gemini AI & running SQL..."):
            sql, result_df, insight = ask_data(prompt)
            
            st.markdown("#### 💻 Generated SQL Query")
            st.code(sql if sql else "-- Could not generate SQL", language="sql")
            
            st.markdown("#### 📊 Database Result")
            if not result_df.empty:
                st.dataframe(result_df, use_container_width=True)
            else:
                st.info("No data returned or query failed.")
                
            st.markdown("#### 💡 Executive Business Insight")
            st.info(insight)

st.markdown("---")
st.caption("🏦 AI Credit Risk Platform • Enterprise UI Edition")
