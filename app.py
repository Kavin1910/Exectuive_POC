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

# Sample data based on worksheet
@st.cache_data
def load_financial_data():
    # Data from July 2025
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

    # Data till August 2025
    august_data = {
        'Company': ['INNOVSPACE', 'INFRASTRIDE', 'TAMALATARIANS', 'SENSE7AI'],
        'Bank_Opening': [42825588, 0, 0, 36991650],  # corrected to match July closing
        'Cash_Opening': [456, 4000000, 0, 0],
        'Loan_Opening': [30000000, 0, 0, 0],
        'Revenue': [16163818, 300101, 0, 28008877],
        'CAPEX_Fixed': [33090055, 1919250, 567855, 59000],
        'CAPEX_Variable': [200000, 309801, 20000, 0],
        'OPEX_Fixed': [12626618, 0, 0, 15774999],
        'OPEX_Variable': [137058, 0, 0, 0],
        'Loan_Closing': [0, 0, 0, 25000000],
        'Bank_Closing': [14717769, 0, 0, 23635549],
        'Cash_Closing': [2806, 400000, 0, 0],
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
        context = f"""
        You are a financial AI assistant for an executive dashboard.
        IMPORTANT: You must only use the provided financial data below. 
        Do NOT use outside knowledge or assumptions.

        User role: {role}

        Financial Data Summary:
        {data.to_string()}

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
    st.title("💰 Executive Financial Dashboard")
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
            st.metric("Total Revenue", f"₹{filtered_df['Revenue'].sum():,.0f}")
        with col2:
            total_capex = filtered_df['CAPEX_Fixed'].sum() + filtered_df['CAPEX_Variable'].sum()
            st.metric("Total CAPEX", f"₹{total_capex:,.0f}")
        with col3:
            total_opex = filtered_df['OPEX_Fixed'].sum() + filtered_df['OPEX_Variable'].sum()
            st.metric("Total OPEX", f"₹{total_opex:,.0f}")
        with col4:
            net_cash = filtered_df['Bank_Closing'].sum() + filtered_df['Cash_Closing'].sum()
            st.metric("Net Cash Position", f"₹{net_cash:,.0f}")

    # Tabs
    if permissions.get('view_company_breakdown', False):
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Company Analysis", "💰 Cash Flow", "📈 Trends", "🤖 AI Insights"])

        with tab1:
            st.header("Company-wise Financial Breakdown")
            chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Stacked Bar", "Treemap"])

            if chart_type == "Bar Chart":
                fig = px.bar(filtered_df, x='Company', y='Revenue', color='Month', title='Revenue by Company')
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Pie Chart":
                revenue_by_company = filtered_df.groupby('Company')['Revenue'].sum()
                fig = px.pie(values=revenue_by_company.values, names=revenue_by_company.index,
                             title='Revenue Distribution')
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Stacked Bar":
                expense_data = filtered_df.melt(
                    id_vars=['Company', 'Month'],
                    value_vars=['CAPEX_Fixed', 'CAPEX_Variable', 'OPEX_Fixed', 'OPEX_Variable'],
                    var_name='Expense_Type', value_name='Amount'
                )
                fig = px.bar(expense_data, x='Company', y='Amount', color='Expense_Type',
                             title='Expense Breakdown by Company')
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Treemap":
                treemap_data = []
                for _, row in filtered_df.iterrows():
                    treemap_data.extend([
                        {'Category': 'Revenue', 'Company': row['Company'], 'Value': row['Revenue']},
                        {'Category': 'CAPEX', 'Company': row['Company'], 'Value': row['CAPEX_Fixed'] + row['CAPEX_Variable']},
                        {'Category': 'OPEX', 'Company': row['Company'], 'Value': row['OPEX_Fixed'] + row['OPEX_Variable']}
                    ])
                treemap_df = pd.DataFrame(treemap_data)
                if not treemap_df.empty:
                    fig = px.treemap(treemap_df, path=['Category', 'Company'], values='Value',
                                     title='Financial Overview Treemap')
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if permissions.get('view_cashflow', False):
                st.header("Cash Flow Analysis")
                cash_flow_data = []
                for _, row in filtered_df.iterrows():
                    opening_cash = row['Bank_Opening'] + row['Cash_Opening']
                    closing_cash = row['Bank_Closing'] + row['Cash_Closing']
                    cash_flow_data.append({
                        'Company': row['Company'],
                        'Month': row['Month'],
                        'Opening_Cash': opening_cash,
                        'Closing_Cash': closing_cash,
                        'Cash_Flow': closing_cash - opening_cash
                    })
                cash_flow_df = pd.DataFrame(cash_flow_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cash_flow_df['Company'], y=cash_flow_df['Opening_Cash'],
                                         mode='lines+markers', name='Opening Cash'))
                fig.add_trace(go.Scatter(x=cash_flow_df['Company'], y=cash_flow_df['Closing_Cash'],
                                         mode='lines+markers', name='Closing Cash'))
                fig.update_layout(title='Cash Position Comparison', xaxis_title='Company', yaxis_title='Amount (₹)')
                st.plotly_chart(fig, use_container_width=True)

                # Format rupees in table
                display_df = cash_flow_df.copy()
                for col in ['Opening_Cash', 'Closing_Cash', 'Cash_Flow']:
                    display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(display_df, use_container_width=True)

        with tab3:
            st.header("Financial Trends")
            if len(filtered_df['Month'].unique()) > 1:
                trend_metrics = ['Revenue', 'CAPEX_Fixed', 'OPEX_Fixed']
                selected_metric = st.selectbox("Select Metric for Trend", trend_metrics)
                fig = px.line(filtered_df, x='Month', y=selected_metric, color='Company',
                              title=f'{selected_metric} Trend Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select multiple months to view trends")

        with tab4:
            st.header("🤖 AI Financial Assistant")
            st.write("Ask questions about your financial data:")
            predefined_questions = [
                "What are the key financial insights from this data?",
                "Which company has the best cash flow position?",
                "What are the major expense categories?",
                "How can we optimize our financial performance?",
                "What trends do you see in our revenue?"
            ]
            selected_question = st.selectbox("Choose a predefined question:", [""] + predefined_questions)
            custom_question = st.text_area("Or ask your own question:")
            question = selected_question if selected_question else custom_question
            if st.button("Get AI Insights") and question:
                with st.spinner("Analyzing financial data..."):
                    insights = get_ai_insights(filtered_df, question, st.session_state.user_role)
                    st.write("**AI Analysis:**")
                    st.write(insights)

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
