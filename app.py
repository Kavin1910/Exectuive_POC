import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from groq import Groq
import json
import os

# Page configuration
st.set_page_config(
    page_title="Executive Financial Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq client (replace API key)
def init_groq():
    try:
        client = Groq(api_key="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")
        return client
    except:
        return None

groq_client = init_groq()

# Corrected sample data based on Excel worksheet - EXACT MATCH
@st.cache_data
def load_financial_data():
    # Data from July 2025 (EXACTLY as per Excel)
    july_data = {
        'Company': ['INNOVSPACE', 'INFRASTRIDE', 'TAMALATARIANS', 'NELLI', 'SENSE7AI'],
        'Bank_Opening': [31059983, 0, 0, 0, 914190],
        'Cash_Opening': [1580, 4000000, 0, 0, 0],
        'Loan_Opening': [80000000, 0, 0, 0, 43500000],
        'Revenue': [12515559, 0, 0, 0, 0],
        'CAPEX_Fixed': [55589933, 48000, 884652, 0, 59000],
        'CAPEX_Variable': [97635, 0, 0, 50000, 0],
        'OPEX_Fixed': [12526675, 0, 0, 0, 15060025],
        'OPEX_Variable': [184183, 0, 0, 0, 0],
        'Loan_Closing': [0, 0, 0, 0, 0],
        'Bank_Closing': [42825588, 4000000, 0, 0, 36991650],
        'Cash_Closing': [456, 0, 0, 0, 0],
        'Month': 'July 2025'
    }

    # Data till August 26, 2025 (EXACTLY as per Excel)
    august_data = {
        'Company': ['INNOVSPACE', 'INFRASTRIDE', 'TAMALATARIANS', 'SENSE7AI'],
        'Bank_Opening': [42825588, 0, 0, 36991650],  # CORRECTED: INFRASTRIDE has 0, SENSE7AI matches July closing
        'Cash_Opening': [456, 400000, 0, 0],  # CORRECTED: INFRASTRIDE has 400000, not 0
        'Loan_Opening': [30000000, 0, 0, 0],  # CORRECTED: Only INNOVSPACE has loan
        'Revenue': [16163318, 300101, 0, 28008877],  # CORRECTED: INNOVSPACE 16163318, INFRASTRIDE 300101, SENSE7AI 28008877
        'CAPEX_Fixed': [33090055, 1919250, 567855, 59000],  # CORRECTED: All values as per Excel
        'CAPEX_Variable': [200000, 309801, 20000, 0],  # CORRECTED: All values as per Excel
        'OPEX_Fixed': [12626618, 0, 0, 15774999],  # CORRECTED: Only INNOVSPACE and SENSE7AI have OPEX
        'OPEX_Variable': [137058, 0, 0, 0],  # CORRECTED: Only INNOVSPACE has variable OPEX
        'Loan_Closing': [25000000, 0, 0, 25000000],  # CORRECTED: Both INNOVSPACE and SENSE7AI have closing loans
        'Bank_Closing': [14717769, 0, 0, 23635549],  # CORRECTED: As per Excel
        'Cash_Closing': [2806, 400000, 0, 0],  # CORRECTED: INFRASTRIDE has 400000
        'Month': 'August 2025'
    }

    df_july = pd.DataFrame(july_data)
    df_august = pd.DataFrame(august_data)

    return pd.concat([df_july, df_august], ignore_index=True)

# Role-based authentication
def authenticate_user():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None

    if not st.session_state.authenticated:
        st.sidebar.title("🔐 Login")

        roles = {
            "accountant": "acc123",
            "finance_manager": "fin456",
            "management": "mgmt789"
        }

        username = st.sidebar.selectbox("Select Role", list(roles.keys()))
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if username in roles and password == roles[username]:
                st.session_state.authenticated = True
                st.session_state.user_role = username
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")

        st.sidebar.markdown("""
        ### Demo Credentials:
        - **Accountant**: accountant / acc123
        - **Finance Manager**: finance_manager / fin456  
        - **Management**: management / mgmt789
        """)
        return False

    return True

# Role permissions
def get_role_permissions(role):
    permissions = {
        'accountant': {
            'view_detailed_financials': True,
            'view_company_breakdown': True,
            'export_data': True,
            'view_cashflow': True,
            'view_summary': False,
            'view_predictions': False
        },
        'finance_manager': {
            'view_detailed_financials': True,
            'view_company_breakdown': True,
            'export_data': True,
            'view_cashflow': True,
            'view_summary': True,
            'view_predictions': True
        },
        'management': {
            'view_detailed_financials': False,
            'view_company_breakdown': True,
            'export_data': True,
            'view_cashflow': True,
            'view_summary': True,
            'view_predictions': True
        }
    }
    return permissions.get(role, {})

# AI assistant (strictly internal data only)
def get_ai_insights(data, question, role):
    if not groq_client:
        return "Groq AI service is not available. Please check your API key configuration."

    try:
        # Format data with rupee symbols for AI context
        formatted_data = data.copy()
        money_columns = ['Bank_Opening', 'Cash_Opening', 'Loan_Opening', 'Revenue', 'CAPEX_Fixed', 
                        'CAPEX_Variable', 'OPEX_Fixed', 'OPEX_Variable', 'Loan_Closing', 'Bank_Closing', 'Cash_Closing']
        
        for col in money_columns:
            if col in formatted_data.columns:
                formatted_data[col] = formatted_data[col].apply(lambda x: f"₹{x:,.0f}")

        context = f"""
        You are a financial AI assistant for an executive dashboard.
        IMPORTANT: You must only use the provided financial data below. 
        Do NOT use outside knowledge or assumptions.
        
        CRITICAL: Always format all monetary values with the rupee symbol (₹) and proper Indian number formatting (e.g., ₹12,51,559 or ₹1,23,45,678).
        When presenting any financial figures, calculations, or analysis, ensure ALL numbers are in rupees with proper formatting.

        User role: {role}

        Financial Data Summary:
        {formatted_data.to_string()}

        Key Instructions for responses:
        1. Always use ₹ symbol before any monetary amount
        2. Use Indian number formatting with commas (lakhs, crores when appropriate)
        3. For large numbers, you can mention crores/lakhs: e.g., ₹1.25 crores for ₹1,25,00,000
        4. Ensure all calculations maintain rupee formatting
        5. When comparing values, always show them in rupees

        Provide insights strictly based on the user's question: {question}
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=800
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

# Main dashboard
def main_dashboard():
    if not authenticate_user():
        st.title("Executive Financial Dashboard")
        st.info("Please login to access the dashboard")
        return

    df = load_financial_data()
    permissions = get_role_permissions(st.session_state.user_role)

    # Header
    st.title("💰 Executive Financial Dashboard - Comprehensive View")
    st.sidebar.success(f"Logged in as: **{st.session_state.user_role.replace('_', ' ').title()}**")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.rerun()

    # Filters
    st.sidebar.header("📊 Filters")
    months = df['Month'].unique()
    selected_month = st.sidebar.selectbox("Select Month", ['All'] + list(months))
    companies = df['Company'].unique()
    selected_companies = st.sidebar.multiselect("Select Companies", companies, default=companies)

    filtered_df = df.copy()
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['Month'] == selected_month]
    filtered_df = filtered_df[filtered_df['Company'].isin(selected_companies)]

    # Executive Summary
    if permissions.get('view_summary', False):
        st.header("📈 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_revenue = filtered_df['Revenue'].sum()
            st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
        with col2:
            total_capex = filtered_df['CAPEX_Fixed'].sum() + filtered_df['CAPEX_Variable'].sum()
            st.metric("Total CAPEX", f"₹{total_capex:,.0f}")
        with col3:
            total_opex = filtered_df['OPEX_Fixed'].sum() + filtered_df['OPEX_Variable'].sum()
            st.metric("Total OPEX", f"₹{total_opex:,.0f}")
        with col4:
            net_cash = filtered_df['Bank_Closing'].sum() + filtered_df['Cash_Closing'].sum()
            st.metric("Net Cash Position", f"₹{net_cash:,.0f}")

    # SINGLE PAGE COMPREHENSIVE CHARTS
    if permissions.get('view_company_breakdown', False):
        st.header("📊 Comprehensive Financial Analysis")
        
        # Create a 2x3 grid of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Revenue by Company Bar Chart
            st.subheader("Revenue by Company")
            revenue_df = filtered_df[filtered_df['Revenue'] > 0]  # Filter out zero revenues
            if not revenue_df.empty:
                fig1 = px.bar(revenue_df, x='Company', y='Revenue', color='Month', 
                             title='Revenue Distribution by Company',
                             labels={'Revenue': 'Revenue (₹)'})
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No revenue data available for selected filters")

        with col2:
            # 2. Revenue Distribution Pie Chart
            st.subheader("Revenue Share Distribution")
            revenue_by_company = filtered_df.groupby('Company')['Revenue'].sum()
            revenue_by_company = revenue_by_company[revenue_by_company > 0]  # Filter out zero revenues
            if not revenue_by_company.empty:
                fig2 = px.pie(values=revenue_by_company.values, names=revenue_by_company.index,
                             title='Company Revenue Share')
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No revenue data available for pie chart")

        # 3. Cash Flow Analysis (Full Width)
        if permissions.get('view_cashflow', False):
            st.subheader("💰 Cash Flow Analysis")
            cash_flow_data = []
            for _, row in filtered_df.iterrows():
                opening_cash = row['Bank_Opening'] + row['Cash_Opening']
                closing_cash = row['Bank_Closing'] + row['Cash_Closing']
                cash_flow_data.append({
                    'Company': row['Company'],
                    'Month': row['Month'],
                    'Opening_Cash': opening_cash,
                    'Closing_Cash': closing_cash,
                    'Cash_Flow': closing_cash - opening_cash,
                    'Revenue': row['Revenue']
                })
            
            cash_flow_df = pd.DataFrame(cash_flow_data)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Cash Position Waterfall Chart
                fig3 = go.Figure()
                companies = cash_flow_df['Company'].unique()
                opening_values = []
                closing_values = []
                
                for company in companies:
                    company_data = cash_flow_df[cash_flow_df['Company'] == company]
                    opening_values.append(company_data['Opening_Cash'].sum())
                    closing_values.append(company_data['Closing_Cash'].sum())
                
                fig3.add_trace(go.Bar(name='Opening Cash', x=companies, y=opening_values,
                                     marker_color='lightblue'))
                fig3.add_trace(go.Bar(name='Closing Cash', x=companies, y=closing_values,
                                     marker_color='darkblue'))
                
                fig3.update_layout(title='Cash Position Comparison',
                                  xaxis_title='Company',
                                  yaxis_title='Amount (₹)',
                                  barmode='group',
                                  height=400)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col4:
                # Cash Flow vs Revenue Scatter Plot
                fig4 = px.scatter(cash_flow_df, x='Revenue', y='Cash_Flow', 
                                 color='Company', size='Closing_Cash',
                                 title='Revenue vs Cash Flow Analysis',
                                 labels={'Cash_Flow': 'Net Cash Flow (₹)', 'Revenue': 'Revenue (₹)'})
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)

        # 4. Expense Breakdown Analysis
        col5, col6 = st.columns(2)
        
        with col5:
            # Stacked Bar Chart for Expenses
            st.subheader("Expense Breakdown by Company")
            expense_data = []
            for _, row in filtered_df.iterrows():
                expense_data.extend([
                    {'Company': row['Company'], 'Month': row['Month'], 'Type': 'CAPEX Fixed', 'Amount': row['CAPEX_Fixed']},
                    {'Company': row['Company'], 'Month': row['Month'], 'Type': 'CAPEX Variable', 'Amount': row['CAPEX_Variable']},
                    {'Company': row['Company'], 'Month': row['Month'], 'Type': 'OPEX Fixed', 'Amount': row['OPEX_Fixed']},
                    {'Company': row['Company'], 'Month': row['Month'], 'Type': 'OPEX Variable', 'Amount': row['OPEX_Variable']}
                ])
            
            expense_df = pd.DataFrame(expense_data)
            expense_df = expense_df[expense_df['Amount'] > 0]  # Filter out zero expenses
            
            if not expense_df.empty:
                fig5 = px.bar(expense_df, x='Company', y='Amount', color='Type',
                             title='Expense Categories by Company')
                fig5.update_layout(height=400)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No expense data available")

        with col6:
            # Financial Trends Over Time
            st.subheader("Financial Trends Over Time")
            if len(filtered_df['Month'].unique()) > 1:
                # Create trend data
                trend_data = []
                for month in filtered_df['Month'].unique():
                    month_data = filtered_df[filtered_df['Month'] == month]
                    trend_data.append({
                        'Month': month,
                        'Total_Revenue': month_data['Revenue'].sum(),
                        'Total_CAPEX': month_data['CAPEX_Fixed'].sum() + month_data['CAPEX_Variable'].sum(),
                        'Total_OPEX': month_data['OPEX_Fixed'].sum() + month_data['OPEX_Variable'].sum()
                    })
                
                trend_df = pd.DataFrame(trend_data)
                
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(x=trend_df['Month'], y=trend_df['Total_Revenue'],
                                         mode='lines+markers', name='Revenue', line=dict(color='green')))
                fig6.add_trace(go.Scatter(x=trend_df['Month'], y=trend_df['Total_CAPEX'],
                                         mode='lines+markers', name='CAPEX', line=dict(color='red')))
                fig6.add_trace(go.Scatter(x=trend_df['Month'], y=trend_df['Total_OPEX'],
                                         mode='lines+markers', name='OPEX', line=dict(color='orange')))
                
                fig6.update_layout(title='Financial Metrics Trend',
                                  xaxis_title='Month',
                                  yaxis_title='Amount (₹)',
                                  height=400)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Select 'All' months to view trends")

        # 5. Treemap for Complete Financial Overview (Full Width)
        st.subheader("🗺️ Financial Overview Treemap")
        treemap_data = []
        for _, row in filtered_df.iterrows():
            if row['Revenue'] > 0:
                treemap_data.append({'Category': 'Revenue', 'Company': row['Company'], 'Value': row['Revenue']})
            if row['CAPEX_Fixed'] + row['CAPEX_Variable'] > 0:
                treemap_data.append({'Category': 'CAPEX', 'Company': row['Company'], 'Value': row['CAPEX_Fixed'] + row['CAPEX_Variable']})
            if row['OPEX_Fixed'] + row['OPEX_Variable'] > 0:
                treemap_data.append({'Category': 'OPEX', 'Company': row['Company'], 'Value': row['OPEX_Fixed'] + row['OPEX_Variable']})
        
        if treemap_data:
            treemap_df = pd.DataFrame(treemap_data)
            fig7 = px.treemap(treemap_df, path=['Category', 'Company'], values='Value',
                             title='Complete Financial Overview',
                             color='Value', color_continuous_scale='RdYlBu')
            fig7.update_layout(height=500)
            st.plotly_chart(fig7, use_container_width=True)

        # AI Insights Section
        st.header("🤖 AI Financial Assistant")
        col7, col8 = st.columns([2, 1])
        
        with col7:
            st.write("Ask questions about your financial data:")
            predefined_questions = [
                "What are the key financial insights from this data?",
                "Which company has the best cash flow position?",
                "What are the major expense categories?",
                "How can we optimize our financial performance?",
                "What trends do you see in our revenue?",
                "Which companies are most capital intensive?",
                "What is the cash burn rate analysis?"
            ]
            selected_question = st.selectbox("Choose a predefined question:", [""] + predefined_questions)
            custom_question = st.text_area("Or ask your own question:")
            question = selected_question if selected_question else custom_question
            
            if st.button("Get AI Insights") and question:
                with st.spinner("Analyzing financial data..."):
                    insights = get_ai_insights(filtered_df, question, st.session_state.user_role)
                    st.write("**AI Analysis:**")
                    st.write(insights)
        
        with col8:
            # Quick Stats
            st.write("**Quick Statistics:**")
            if not filtered_df.empty:
                stats = {
                    "Active Companies": len(filtered_df['Company'].unique()),
                    "Months Covered": len(filtered_df['Month'].unique()),
                    "Total Records": len(filtered_df),
                    "Avg Revenue": f"₹{filtered_df['Revenue'].mean():,.0f}",
                    "Highest Revenue": f"₹{filtered_df['Revenue'].max():,.0f}",
                    "Total Cash Flow": f"₹{(filtered_df['Bank_Closing'] + filtered_df['Cash_Closing'] - filtered_df['Bank_Opening'] - filtered_df['Cash_Opening']).sum():,.0f}"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)

    # Detailed Financial Data
    if permissions.get('view_detailed_financials', False):
        st.header("📋 Detailed Financial Data")
        df_display = filtered_df.copy()
        money_cols = [c for c in df_display.columns if any(x in c for x in
                     ['Opening', 'Closing', 'Revenue', 'CAPEX', 'OPEX', 'Loan'])]
        for col in money_cols:
            df_display[col] = df_display[col].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(df_display, use_container_width=True)

    # Export options
    if permissions.get('export_data', False):
        st.sidebar.header("📥 Export Data")
        if st.sidebar.button("Generate Excel Export"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Financial_Data', index=False)
                if permissions.get('view_summary', False):
                    summary_data = {
                        'Metric': ['Total Revenue', 'Total CAPEX', 'Total OPEX', 'Net Cash'],
                        'Amount': [
                            filtered_df['Revenue'].sum(),
                            filtered_df['CAPEX_Fixed'].sum() + filtered_df['CAPEX_Variable'].sum(),
                            filtered_df['OPEX_Fixed'].sum() + filtered_df['OPEX_Variable'].sum(),
                            filtered_df['Bank_Closing'].sum() + filtered_df['Cash_Closing'].sum()
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            excel_data = output.getvalue()
            st.sidebar.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main_dashboard()
