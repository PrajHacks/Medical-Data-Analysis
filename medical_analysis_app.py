import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, chi2, linregress
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- Common Page Configuration ---
st.set_page_config(layout="wide", page_title="Medical Data Analysis Hub 🩺")

# --- Home Page ---
def home_page():
    st.title("StatEase - Medical Data Analysis Hub")
    st.markdown("""
    Welcome to your user-friendly statistical analysis software designed for medical professionals.
    This tool helps you analyze clinical data with ease, even with minimal statistical and coding knowledge.
    """)

    st.markdown("---")  # Visual separator

    st.header("1. Choose Your Analysis Type")
    st.write("Select the specialized analysis module you wish to access:")
    analysis_type = st.radio(
        "Select the type of analysis you want to perform:",
        ("Meta-Analysis", "Observational Data Analysis"),
        key="analysis_type_selection",
        horizontal=True
    )

    # Reset sample data if analysis type changes
    if 'previous_analysis_type' not in st.session_state:
        st.session_state.previous_analysis_type = analysis_type

    if st.session_state.previous_analysis_type != analysis_type:
        if 'sample_loaded' in st.session_state:
            del st.session_state.sample_loaded
        if 'sample_file_path' in st.session_state:
            del st.session_state.sample_file_path
        if 'df_initial_preview' in st.session_state:
            del st.session_state.df_initial_preview
        st.session_state.previous_analysis_type = analysis_type

    st.markdown("---")  # Visual separator

    st.header("2. Upload Your Data File or Use Sample Data")
    st.write("Please upload your dataset (CSV or Excel formats) or select 'Use Sample Data' to proceed with predefined datasets.")
    
    # Create two columns for file uploader and sample data button
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV or Excel file (XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            key="main_file_uploader"
        )

    with col2:
        # Add some vertical space to push button down
        st.write("")  
        st.write("")  
        st.write("")  
        # Only show button if sample data hasn't been loaded yet
        if 'sample_loaded' not in st.session_state or not st.session_state.sample_loaded:
            use_sample_data = st.button("Use Sample Data", key="use_sample_data_button")
            if use_sample_data:
                if analysis_type == "Meta-Analysis":
                    sample_file_path = os.path.join(os.path.dirname(__file__), "Rabies.xlsx")
                elif analysis_type == "Observational Data Analysis":
                    sample_file_path = os.path.join(os.path.dirname(__file__), "tea.xlsx")
                
                if sample_file_path and os.path.exists(sample_file_path):
                    st.session_state.sample_loaded = True
                    st.session_state.sample_file_path = sample_file_path
                    st.success(f"Loaded sample data: {os.path.basename(sample_file_path)}")
                else:
                    st.error(f"Sample data file {sample_file_path} not found in the script directory.")
                    uploaded_file = None
        else:
            st.write(f"Sample data loaded: {os.path.basename(st.session_state.sample_file_path)}")

    # Determine the file to use
    sample_file_path = st.session_state.sample_file_path if 'sample_loaded' in st.session_state and st.session_state.sample_loaded else None

    return analysis_type, uploaded_file, sample_file_path

# --- Meta-Analysis Application Logic ---
def run_meta_analysis_app(df_raw):
    st.title("📊 Meta-Analysis & Forest Plot Visualization")
    st.markdown("""
    This section allows you to visualize the proportion of RVNA >0.5IU/L or Adverse Events
    for Intervention and Control groups across different studies, and perform meta-analysis.
    """)

    st.markdown("---")  # Visual separator

    # --- Data Preprocessing for Meta-Analysis ---
    meta_analysis_column_names_list = [
        'Study',
        'RVNA_0d_Intervention_Count', 'RVNA_0d_Intervention_Population',
        'RVNA_0d_Control_Count', 'RVNA_0d_Control_Population',
        'RVNA_14d_Intervention_Count', 'RVNA_14d_Intervention_Population',
        'RVNA_14d_Control_Count', 'RVNA_14d_Control_Population',
        'RVNA_42d_Intervention_Count', 'RVNA_42d_Intervention_Population',
        'RVNA_42d_Control_Count', 'RVNA_42d_Control_Population',
        'RVNA_82d_Intervention_Count', 'RVNA_82d_Intervention_Population',
        'RVNA_82d_Control_Count', 'RVNA_82d_Control_Population',
        'All_AE_Intervention_Count', 'All_AE_Intervention_Population',
        'All_AE_Control_Count', 'All_AE_Control_Population',
        'Local_AE_Intervention_Count', 'Local_AE_Intervention_Population',
        'Local_AE_Control_Count', 'Local_AE_Control_Population'
    ]

    df_meta = df_raw.copy()

    expected_num_cols_meta = len(meta_analysis_column_names_list)
    if len(df_meta.columns) != expected_num_cols_meta:
        st.error(f"**Data Format Mismatch for Meta-Analysis:** The uploaded file does not seem to have the expected number of columns for Meta-Analysis. "
                 f"Expected {expected_num_cols_meta} columns, but found {len(df_meta.columns)}. "
                 "Please ensure your file matches the required data structure for Meta-Analysis "
                 "(e.g., 'Rabies.xlsx - Sheet2.csv').")
        st.stop()

    df_meta.columns = meta_analysis_column_names_list

    numeric_cols_meta = [col for col in df_meta.columns if 'Count' in col or 'Population' in col]
    for col in numeric_cols_meta:
        df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce')

    df_meta['Intervention_Proportion_0d'] = df_meta['RVNA_0d_Intervention_Count'] / df_meta['RVNA_0d_Intervention_Population']
    df_meta['Control_Proportion_0d'] = df_meta['RVNA_0d_Control_Count'] / df_meta['RVNA_0d_Control_Population']
    df_meta['Intervention_Proportion_14d'] = df_meta['RVNA_14d_Intervention_Count'] / df_meta['RVNA_14d_Intervention_Population']
    df_meta['Control_Proportion_14d'] = df_meta['RVNA_14d_Control_Count'] / df_meta['RVNA_14d_Control_Population']
    df_meta['Intervention_Proportion_42d'] = df_meta['RVNA_42d_Intervention_Count'] / df_meta['RVNA_42d_Intervention_Population']
    df_meta['Control_Proportion_42d'] = df_meta['RVNA_42d_Control_Count'] / df_meta['RVNA_42d_Control_Population']
    df_meta['Intervention_Proportion_82d'] = df_meta['RVNA_82d_Intervention_Count'] / df_meta['RVNA_82d_Intervention_Population']
    df_meta['Control_Proportion_82d'] = df_meta['RVNA_82d_Control_Count'] / df_meta['RVNA_82d_Control_Population']
    df_meta['All_AE_Intervention_Proportion'] = df_meta['All_AE_Intervention_Count'] / df_meta['All_AE_Intervention_Population']
    df_meta['All_AE_Control_Proportion'] = df_meta['All_AE_Control_Count'] / df_meta['All_AE_Control_Population']
    df_meta['Local_AE_Intervention_Proportion'] = df_meta['Local_AE_Intervention_Count'] / df_meta['Local_AE_Intervention_Population']
    df_meta['Local_AE_Control_Proportion'] = df_meta['Local_AE_Control_Count'] / df_meta['Local_AE_Control_Population']

    def calculate_risk_ratio_and_ci(df_input, outcome_prefix):
        df_calc = df_input.copy()
        int_count_col = f'{outcome_prefix}_Intervention_Count'
        int_pop_col = f'{outcome_prefix}_Intervention_Population'
        ctrl_count_col = f'{outcome_prefix}_Control_Count'
        ctrl_pop_col = f'{outcome_prefix}_Control_Population'

        for col in [int_count_col, int_pop_col, ctrl_count_col, ctrl_pop_col]:
            if col not in df_calc.columns:
                st.error(f"Missing expected column for Forest Plot calculation: {col}")
                return pd.DataFrame()
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

        df_calc['a'] = df_calc[int_count_col]
        df_calc['n1'] = df_calc[int_pop_col]
        df_calc['c'] = df_calc[ctrl_count_col]
        df_calc['n0'] = df_calc[ctrl_pop_col]

        df_calc['RR'] = np.nan
        df_calc['RR_Lower_CI'] = np.nan
        df_calc['RR_Upper_CI'] = np.nan
        df_calc['log_RR'] = np.nan
        df_calc['V_log_RR'] = np.nan

        valid_rows_mask = (df_calc['n1'] > 0) & (df_calc['n0'] > 0)
        df_valid = df_calc[valid_rows_mask].copy()

        if not df_valid.empty:
            df_valid['a_corr'] = df_valid['a'].apply(lambda x: x + 0.5 if x == 0 else x)
            df_valid['n1_corr'] = df_valid['n1'].apply(lambda x: x + 0.5 if x == 0 else x)
            df_valid['c_corr'] = df_valid['c'].apply(lambda x: x + 0.5 if x == 0 else x)
            df_valid['n0_corr'] = df_valid['n0'].apply(lambda x: x + 0.5 if x == 0 else x)

            df_valid['p1'] = df_valid['a_corr'] / df_valid['n1_corr']
            df_valid['p0'] = df_valid['c_corr'] / df_valid['n0_corr']

            df_valid['RR'] = df_valid['p1'] / df_valid['p0']
            df_valid['log_RR'] = np.log(df_valid['RR'])

            df_valid['SE_log_RR'] = np.sqrt(
                (1 - df_valid['p1']) / df_valid['a_corr'] +
                (1 - df_valid['p0']) / df_valid['c_corr']
            )
            df_valid['V_log_RR'] = df_valid['SE_log_RR']**2

            z_score = norm.ppf(0.975)
            df_valid['log_RR_lower'] = df_valid['log_RR'] - z_score * df_valid['SE_log_RR']
            df_valid['log_RR_upper'] = df_valid['log_RR'] + z_score * df_valid['SE_log_RR']

            df_valid['RR_Lower_CI'] = np.exp(df_valid['log_RR_lower'])
            df_valid['RR_Upper_CI'] = np.exp(df_valid['log_RR_upper'])

            df_calc.loc[valid_rows_mask, ['RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']] = \
                df_valid[['RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']]
        return df_calc[['Study', 'a', 'n1', 'c', 'n0', 'RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']]

    def perform_meta_analysis(forest_df):
        df_for_meta = forest_df.dropna(subset=['log_RR', 'V_log_RR']).copy()
        if df_for_meta.empty:
            return None

        df_for_meta['w_fixed'] = 1 / df_for_meta['V_log_RR']
        sum_w_fixed = df_for_meta['w_fixed'].sum()

        fixed_weights = pd.Series(np.nan, index=df_for_meta.index)
        log_RR_fixed_overall = np.nan
        SE_log_RR_fixed_overall = np.nan
        RR_fixed_overall = np.nan
        RR_fixed_lower_ci = np.nan
        RR_fixed_upper_ci = np.nan

        if sum_w_fixed > 0:
            fixed_weights = df_for_meta['w_fixed'] / sum_w_fixed * 100
            log_RR_fixed_overall = (df_for_meta['w_fixed'] * df_for_meta['log_RR']).sum() / sum_w_fixed
            SE_log_RR_fixed_overall = np.sqrt(1 / sum_w_fixed)
            RR_fixed_overall = np.exp(log_RR_fixed_overall)
            RR_fixed_lower_ci = np.exp(log_RR_fixed_overall - norm.ppf(0.975) * SE_log_RR_fixed_overall)
            RR_fixed_upper_ci = np.exp(log_RR_fixed_overall + norm.ppf(0.975) * SE_log_RR_fixed_overall)

        Q = np.nan
        if pd.notna(log_RR_fixed_overall):
            Q = (df_for_meta['w_fixed'] * (df_for_meta['log_RR'] - log_RR_fixed_overall)**2).sum()

        df_heterogeneity = len(df_for_meta) - 1
        if df_heterogeneity < 0: df_heterogeneity = 0

        p_heterogeneity = 1.0
        if df_heterogeneity > 0 and pd.notna(Q):
            p_heterogeneity = 1 - chi2.cdf(Q, df_heterogeneity)

        I2 = 0.0
        if pd.notna(Q) and Q > 0 and df_heterogeneity > 0:
            I2 = max(0, ((Q - df_heterogeneity) / Q) * 100)

        tau_squared = 0.0
        C = sum_w_fixed - (df_for_meta['w_fixed']**2).sum() / sum_w_fixed if sum_w_fixed > 0 else 0
        if C > 0 and pd.notna(Q) and df_heterogeneity > 0:
            tau_squared = max(0, (Q - df_heterogeneity) / C)

        df_for_meta['w_random'] = 1 / (df_for_meta['V_log_RR'] + tau_squared)
        sum_w_random = df_for_meta['w_random'].sum()

        random_weights = pd.Series(np.nan, index=df_for_meta.index)
        log_RR_random_overall = np.nan
        SE_log_RR_random_overall = np.nan
        RR_random_overall = np.nan
        RR_random_lower_ci = np.nan
        RR_random_upper_ci = np.nan

        if sum_w_random > 0:
            random_weights = df_for_meta['w_random'] / sum_w_random * 100
            log_RR_random_overall = (df_for_meta['w_random'] * df_for_meta['log_RR']).sum() / sum_w_random
            SE_log_RR_random_overall = np.sqrt(1 / sum_w_random)
            RR_random_overall = np.exp(log_RR_random_overall)
            RR_random_lower_ci = np.exp(log_RR_random_overall - norm.ppf(0.975) * SE_log_RR_random_overall)
            RR_random_upper_ci = np.exp(log_RR_random_overall + norm.ppf(0.975) * SE_log_RR_random_overall)

        total_int_events = df_for_meta['a'].sum()
        total_int_pop = df_for_meta['n1'].sum()
        total_ctrl_events = df_for_meta['c'].sum()
        total_ctrl_pop = df_for_meta['n0'].sum()

        return {
            'fixed_effect': {
                'RR': RR_fixed_overall,
                'Lower_CI': RR_fixed_lower_ci,
                'Upper_CI': RR_fixed_upper_ci,
                'Total_Events_Int': total_int_events,
                'Total_Pop_Int': total_int_pop,
                'Total_Events_Ctrl': total_ctrl_events,
                'Total_Pop_Ctrl': total_ctrl_pop,
            },
            'random_effects': {
                'RR': RR_random_overall,
                'Lower_CI': RR_random_lower_ci,
                'Upper_CI': RR_random_upper_ci,
            },
            'heterogeneity': {
                'Q': Q,
                'df': df_heterogeneity,
                'p_value': p_heterogeneity,
                'I2': I2,
                'tau_squared': tau_squared
            },
            'study_weights': {
                'fixed': fixed_weights,
                'random': random_weights
            }
        }

    tab_titles = ["Statistical Analysis", "Risk of Bias", "Sensitivity Analysis", "Baujat Plot", "Funnel Plot"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Core Statistical Analysis")
        st.write("Explore different visualizations and regression for your meta-analysis data.")
        st.markdown("---")

        plot_type_meta = st.radio(
            "Choose Plot Type for Statistical Analysis:",
            ("Bar Graph", "Forest Plot", "RVNA Proportion vs. Days Regression"),
            key="meta_plot_type",
            horizontal=True
        )

        if plot_type_meta != "RVNA Proportion vs. Days Regression":
            graph_option_meta = st.selectbox(
                "Select data to visualize:",
                (
                    "RVNA >0.5IU/L at 0 days",
                    "RVNA >0.5IU/L at 14 days",
                    "RVNA >0.5IU/L at 42 days",
                    "RVNA >0.5IU/L at 82 days",
                    "All Adverse Events",
                    "Local Adverse Events"
                ),
                key="meta_graph_option"
            )
            st.markdown("---")

        if plot_type_meta == "Bar Graph":
            df_plot = pd.DataFrame()
            title = ""
            y_axis_label = "Proportion"

            if graph_option_meta == "RVNA >0.5IU/L at 0 days":
                df_plot = df_meta[['Study', 'Intervention_Proportion_0d', 'Control_Proportion_0d']].copy()
                df_plot = df_plot.rename(columns={
                    'Intervention_Proportion_0d': 'Intervention',
                    'Control_Proportion_0d': 'Control'
                })
                title = 'Proportion of RVNA >0.5IU/L at 0 Days: Intervention vs. Control'
                y_axis_label = 'Proportion of RVNA >0.5IU/L'
            elif graph_option_meta == "RVNA >0.5IU/L at 14 days":
                df_plot = df_meta[['Study', 'Intervention_Proportion_14d', 'Control_Proportion_14d']].copy()
                df_plot = df_plot.rename(columns={
                    'Intervention_Proportion_14d': 'Intervention',
                    'Control_Proportion_14d': 'Control'
                })
                title = 'Proportion of RVNA >0.5IU/L at 14 Days: Intervention vs. Control'
                y_axis_label = 'Proportion of RVNA >0.5IU/L'
            elif graph_option_meta == "RVNA >0.5IU/L at 42 days":
                df_plot = df_meta[['Study', 'Intervention_Proportion_42d', 'Control_Proportion_42d']].copy()
                df_plot = df_plot.rename(columns={
                    'Intervention_Proportion_42d': 'Intervention',
                    'Control_Proportion_42d': 'Control'
                })
                title = 'Proportion of RVNA >0.5IU/L at 42 Days: Intervention vs. Control'
                y_axis_label = 'Proportion of RVNA >0.5IU/L'
            elif graph_option_meta == "RVNA >0.5IU/L at 82 days":
                df_plot = df_meta[['Study', 'Intervention_Proportion_82d', 'Control_Proportion_82d']].copy()
                df_plot = df_plot.rename(columns={
                    'Intervention_Proportion_82d': 'Intervention',
                    'Control_Proportion_82d': 'Control'
                })
                title = 'Proportion of RVNA >0.5IU/L at 82 Days: Intervention vs. Control'
                y_axis_label = 'Proportion of RVNA >0.5IU/L'
            elif graph_option_meta == "All Adverse Events":
                df_plot = df_meta[['Study', 'All_AE_Intervention_Proportion', 'All_AE_Control_Proportion']].copy()
                df_plot = df_plot.rename(columns={
                    'All_AE_Intervention_Proportion': 'Intervention',
                    'All_AE_Control_Proportion': 'Control'
                })
                title = 'Proportion of All Adverse Events: Intervention vs. Control'
                y_axis_label = 'Proportion of All Adverse Events'
            elif graph_option_meta == "Local Adverse Events":
                df_plot = df_meta[['Study', 'Local_AE_Intervention_Proportion', 'Local_AE_Control_Proportion']].copy()
                df_plot = df_plot.rename(columns={
                    'Local_AE_Intervention_Proportion': 'Intervention',
                    'Local_AE_Control_Proportion': 'Control'
                })
                title = 'Proportion of Local Adverse Events: Intervention vs. Control'
                y_axis_label = 'Proportion of Local Adverse Events'

            df_melted = df_plot.melt(id_vars=['Study'], var_name='Group', value_name='Proportion')
            df_melted.dropna(subset=['Proportion'], inplace=True)

            if not df_melted.empty:
                fig = px.bar(
                    df_melted,
                    x='Study',
                    y='Proportion',
                    color='Group',
                    barmode='group',
                    title=title,
                    labels={'Proportion': y_axis_label, 'Study': 'Study'},
                    color_discrete_map={'Intervention': '#4CAF50', 'Control': '#FFC107'}
                )
                fig.update_layout(
                    xaxis_title="Study",
                    yaxis_title=y_axis_label,
                    xaxis_tickangle=-45,
                    yaxis_range=[0, 1.1],
                    legend_title_text='Group'
                )
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available to plot for the selected visualization option after processing.")

        elif plot_type_meta == "Forest Plot":
            outcome_prefix_map = {
                "RVNA >0.5IU/L at 0 days": "RVNA_0d", "RVNA >0.5IU/L at 14 days": "RVNA_14d",
                "RVNA >0.5IU/L at 42 days": "RVNA_42d", "RVNA >0.5IU/L at 82 days": "RVNA_82d",
                "All Adverse Events": "All_AE", "Local Adverse Events": "Local_AE"
            }
            selected_outcome_prefix = outcome_prefix_map.get(graph_option_meta)

            if selected_outcome_prefix:
                forest_df = calculate_risk_ratio_and_ci(df_meta, selected_outcome_prefix)
                if not forest_df.empty:
                    meta_results = perform_meta_analysis(forest_df)
                    st.subheader(f"Forest Plot: Risk Ratio for {graph_option_meta}")
                    st.write("This plot visualizes the effect size (Risk Ratio) and its confidence interval for each study, along with the overall meta-analysis results.")

                    fig_forest = go.Figure()
                    forest_df_plot = forest_df.dropna(subset=['RR', 'RR_Lower_CI', 'RR_Upper_CI']).copy()

                    fig_forest.add_trace(go.Scatter(
                        x=forest_df_plot['RR'],
                        y=forest_df_plot['Study'],
                        mode='markers',
                        marker=dict(color='blue', size=10, symbol='square'),
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=forest_df_plot['RR_Upper_CI'] - forest_df_plot['RR'],
                            arrayminus=forest_df_plot['RR'] - forest_df_plot['RR_Lower_CI'],
                            color='blue',
                            thickness=1.5,
                            width=5
                        ),
                        name='Individual Studies'
                    ))

                    if meta_results:
                        if pd.notna(meta_results['fixed_effect']['RR']):
                            fig_forest.add_trace(go.Scatter(
                                x=[meta_results['fixed_effect']['RR']],
                                y=["Overall (Fixed-effect)"],
                                mode='markers',
                                marker=dict(
                                    symbol='diamond',
                                    size=20,
                                    color='darkgreen',
                                    line=dict(width=1, color='darkgreen')
                                ),
                                error_x=dict(
                                    type='data',
                                    symmetric=False,
                                    array=[meta_results['fixed_effect']['Upper_CI'] - meta_results['fixed_effect']['RR']],
                                    arrayminus=[meta_results['fixed_effect']['RR'] - meta_results['fixed_effect']['Lower_CI']],
                                    color='darkgreen',
                                    thickness=2,
                                    width=10
                                ),
                                name='Overall (Fixed-effect)'
                            ))

                        if pd.notna(meta_results['random_effects']['RR']):
                            fig_forest.add_trace(go.Scatter(
                                x=[meta_results['random_effects']['RR']],
                                y=["Overall (Random-effects)"],
                                mode='markers',
                                marker=dict(
                                    symbol='diamond',
                                    size=20,
                                    color='purple',
                                    line=dict(width=1, color='purple')
                                ),
                                error_x=dict(
                                    type='data',
                                    symmetric=False,
                                    array=[meta_results['random_effects']['Upper_CI'] - meta_results['random_effects']['RR']],
                                    arrayminus=[meta_results['random_effects']['RR'] - meta_results['random_effects']['Lower_CI']],
                                    color='purple',
                                    thickness=2,
                                    width=10
                                ),
                                name='Overall (Random-effects)'
                            ))

                    fig_forest.add_shape(
                        type="line",
                        x0=1, x1=1, y0=0, y1=1,
                        xref='x', yref='paper',
                        line=dict(color="red", width=1, dash="dash"),
                    )

                    fig_forest.add_annotation(
                        x=1, y=1.02,
                        xref="x", yref="paper",
                        text="No Effect",
                        showarrow=False,
                        font=dict(color="red", size=10),
                        xanchor="left"
                    )

                    fig_forest.update_layout(
                        title=f'Forest Plot: Risk Ratio for {graph_option_meta}',
                        xaxis_title='Risk Ratio (Intervention vs. Control)',
                        yaxis_title='Study',
                        yaxis_autorange="reversed",
                        hovermode="y unified",
                        height=600,
                        showlegend=False
                    )

                    y_axis_categories = forest_df_plot['Study'].tolist()
                    if meta_results and pd.notna(meta_results['fixed_effect']['RR']):
                        y_axis_categories.append("Overall (Fixed-effect)")
                    if meta_results and pd.notna(meta_results['random_effects']['RR']):
                        y_axis_categories.append("Overall (Random-effects)")
                    fig_forest.update_yaxes(categoryorder='array', categoryarray=y_axis_categories)

                    valid_rrs = forest_df_plot['RR_Lower_CI'].dropna().tolist() + \
                                forest_df_plot['RR_Upper_CI'].dropna().tolist()
                    if meta_results and pd.notna(meta_results['fixed_effect']['Lower_CI']):
                        valid_rrs.append(meta_results['fixed_effect']['Lower_CI'])
                        valid_rrs.append(meta_results['fixed_effect']['Upper_CI'])
                    if meta_results and pd.notna(meta_results['random_effects']['Lower_CI']):
                        valid_rrs.append(meta_results['random_effects']['Lower_CI'])
                        valid_rrs.append(meta_results['random_effects']['Upper_CI'])

                    if valid_rrs:
                        min_rr_data = min(valid_rrs)
                        max_rr_data = max(valid_rrs)
                        min_rr = min(min_rr_data * 0.9, 0.9)
                        max_rr = max(max_rr_data * 1.1, 1.1)
                    else:
                        min_rr, max_rr = 0.5, 1.5

                    fig_forest.update_xaxes(range=[min_rr, max_rr])
                    st.plotly_chart(fig_forest, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Detailed Study Data and Meta-Analysis Results")
                    st.write("This table provides a comprehensive overview of each study's contribution and the overall meta-analysis findings.")

                    if meta_results:
                        table_data = forest_df[['Study', 'a', 'n1', 'c', 'n0', 'RR', 'RR_Lower_CI', 'RR_Upper_CI']].copy()
                        table_data['RR (95% CI)'] = table_data.apply(
                            lambda row: f"{row['RR']:.2f} [{row['RR_Lower_CI']:.2f}, {row['RR_Upper_CI']:.2f}]"
                                        if pd.notna(row['RR']) and pd.notna(row['RR_Lower_CI']) and pd.notna(row['RR_Upper_CI'])
                                        else "N/A", axis=1
                        )

                        table_data['Weight (Fixed)'] = ""
                        table_data['Weight (Random)'] = ""

                        if not meta_results['study_weights']['fixed'].empty:
                            for meta_idx, study_row in forest_df_plot.iterrows():
                                original_df_idx = forest_df[forest_df['Study'] == study_row['Study']].index
                                if not original_df_idx.empty:
                                    weight = meta_results['study_weights']['fixed'].get(meta_idx)
                                    if pd.notna(weight):
                                        table_data.loc[original_df_idx[0], 'Weight (Fixed)'] = f"{weight:.1f}%"
                                    else:
                                        table_data.loc[original_df_idx[0], 'Weight (Fixed)'] = "N/A"

                        if not meta_results['study_weights']['random'].empty:
                            for meta_idx, study_row in forest_df_plot.iterrows():
                                original_df_idx = forest_df[forest_df['Study'] == study_row['Study']].index
                                if not original_df_idx.empty:
                                    weight = meta_results['study_weights']['random'].get(meta_idx)
                                    if pd.notna(weight):
                                        table_data.loc[original_df_idx[0], 'Weight (Random)'] = f"{weight:.1f}%"
                                    else:
                                        table_data.loc[original_df_idx[0], 'Weight (Random)'] = "N/A"

                        overall_fixed_row = {
                            'Study': "Overall (Fixed-effect)",
                            'a': int(meta_results['fixed_effect']['Total_Events_Int']) if pd.notna(meta_results['fixed_effect']['Total_Events_Int']) else "",
                            'n1': int(meta_results['fixed_effect']['Total_Pop_Int']) if pd.notna(meta_results['fixed_effect']['Total_Pop_Int']) else "",
                            'c': int(meta_results['fixed_effect']['Total_Events_Ctrl']) if pd.notna(meta_results['fixed_effect']['Total_Events_Ctrl']) else "",
                            'n0': int(meta_results['fixed_effect']['Total_Pop_Ctrl']) if pd.notna(meta_results['fixed_effect']['Total_Pop_Ctrl']) else "",
                            'RR': meta_results['fixed_effect']['RR'].round(2) if pd.notna(meta_results['fixed_effect']['RR']) else np.nan,
                            'RR_Lower_CI': meta_results['fixed_effect']['Lower_CI'].round(2) if pd.notna(meta_results['fixed_effect']['Lower_CI']) else np.nan,
                            'RR_Upper_CI': meta_results['fixed_effect']['Upper_CI'].round(2) if pd.notna(meta_results['fixed_effect']['Upper_CI']) else np.nan,
                            'RR (95% CI)': f"{meta_results['fixed_effect']['RR']:.2f} [{meta_results['fixed_effect']['Lower_CI']:.2f}, {meta_results['fixed_effect']['Upper_CI']:.2f}]" if pd.notna(meta_results['fixed_effect']['RR']) else "N/A",
                            'Weight (Fixed)': "100.0%", 'Weight (Random)': ""
                        }
                        overall_random_row = {
                            'Study': "Overall (Random-effects)",
                            'a': "", 'n1': "", 'c': "", 'n0': "",
                            'RR': meta_results['random_effects']['RR'].round(2) if pd.notna(meta_results['random_effects']['RR']) else np.nan,
                            'RR_Lower_CI': meta_results['random_effects']['Lower_CI'].round(2) if pd.notna(meta_results['random_effects']['Lower_CI']) else np.nan,
                            'RR_Upper_CI': meta_results['random_effects']['Upper_CI'].round(2) if pd.notna(meta_results['random_effects']['Upper_CI']) else np.nan,
                            'RR (95% CI)': f"{meta_results['random_effects']['RR']:.2f} [{meta_results['random_effects']['Lower_CI']:.2f}, {meta_results['random_effects']['Upper_CI']:.2f}]" if pd.notna(meta_results['random_effects']['RR']) else "N/A",
                            'Weight (Fixed)': "", 'Weight (Random)': "100.0%"
                        }

                        table_data = pd.concat([table_data, pd.DataFrame([overall_fixed_row, overall_random_row])], ignore_index=True)
                        table_data.rename(columns={
                            'a': 'Events (Exp)',
                            'n1': 'Total (Exp)',
                            'c': 'Events (Ctrl)',
                            'n0': 'Total (Ctrl)'
                        }, inplace=True)

                        display_cols = [
                            'Study', 'Events (Exp)', 'Total (Exp)',
                            'Events (Ctrl)', 'Total (Ctrl)',
                            'RR (95% CI)', 'Weight (Fixed)', 'Weight (Random)'
                        ]
                        st.dataframe(table_data[display_cols].fillna(''), use_container_width=True)

                        st.markdown("---")
                        st.subheader("Heterogeneity Statistics")
                        st.write("These metrics quantify the variability among study results.")
                        st.markdown(f"**Heterogeneity:** $I^2 = {meta_results['heterogeneity']['I2']:.1f}\\%$, "
                                    f"$\\tau^2 = {meta_results['heterogeneity']['tau_squared']:.2f}$, "
                                    f"p-value = {meta_results['heterogeneity']['p_value']:.3f}")
                        st.markdown("""
                        * **$I^2$ (I-squared):** Describes the percentage of total variation across studies that is due to heterogeneity rather than chance.
                        * **$\\tau^2$ (Tau-squared):** Estimates the variance of the true effect sizes across studies in a random-effects model.
                        * **p-value:** For Cochran's Q test, a low p-value (e.g., < 0.10) suggests significant heterogeneity.
                        """)
                    else:
                        st.warning(f"Could not perform meta-analysis for '{graph_option_meta}'.")
                else:
                    st.warning(f"Could not generate Forest Plot for '{graph_option_meta}'.")
            else:
                st.error("Invalid graph option selected for Forest Plot.")

        elif plot_type_meta == "RVNA Proportion vs. Days Regression":
            st.subheader("Regression Analysis: RVNA Proportion vs. Days (Per Study)")
            st.write("This analysis explores the linear relationship between RVNA proportion and time (in days) for each individual study.")
            st.markdown("---")

            regression_group_meta = st.selectbox(
                "Select group for RVNA Proportion vs. Days Regression:",
                ("Intervention", "Control"),
                key="meta_regression_group"
            )
            st.markdown("---")

            expected_proportion_cols_for_group_meta = [
                f'{regression_group_meta}_Proportion_0d',
                f'{regression_group_meta}_Proportion_14d',
                f'{regression_group_meta}_Proportion_42d',
                f'{regression_group_meta}_Proportion_82d'
            ]

            actual_proportion_cols_meta = [col for col in expected_proportion_cols_for_group_meta if col in df_meta.columns]

            if not actual_proportion_cols_meta:
                st.warning(f"No RVNA proportion data found for {regression_group_meta} to perform regression analysis.")
            else:
                df_melt_regression_meta = df_meta[['Study'] + actual_proportion_cols_meta].copy()
                df_long_regression_meta = df_melt_regression_meta.melt(
                    id_vars=['Study'],
                    value_vars=actual_proportion_cols_meta,
                    var_name='Proportion_Type',
                    value_name='Proportion'
                )

                day_mapping_meta = {
                    f'{regression_group_meta}_Proportion_0d': 0,
                    f'{regression_group_meta}_Proportion_14d': 14,
                    f'{regression_group_meta}_Proportion_42d': 42,
                    f'{regression_group_meta}_Proportion_82d': 82,
                }

                df_long_regression_meta['Days'] = df_long_regression_meta['Proportion_Type'].map(day_mapping_meta)
                df_long_regression_meta.dropna(subset=['Proportion', 'Days'], inplace=True)

                if df_long_regression_meta.empty:
                    st.warning(f"No valid data points available for regression analysis for {regression_group_meta} after processing.")
                else:
                    fig_regression_meta = go.Figure()
                    regression_results_list_meta = []
                    for study in df_long_regression_meta['Study'].unique():
                        study_data = df_long_regression_meta[df_long_regression_meta['Study'] == study]
                        fig_regression_meta.add_trace(go.Scatter(
                            x=study_data['Days'], y=study_data['Proportion'], mode='markers',
                            name=f'{study} Data', marker=dict(size=8)
                        ))
                        if len(study_data['Days'].unique()) >= 2:
                            x = study_data['Days']
                            y = study_data['Proportion']
                            slope, intercept, r_value, p_value, std_err = linregress(x, y)
                            fig_regression_meta.add_trace(go.Scatter(
                                x=x, y=slope * x + intercept, mode='lines',
                                name=f'{study} Trend', line=dict(dash='dash')
                            ))
                            regression_results_list_meta.append({
                                'Study': study, 'Slope': f"{slope:.3f}", 'Intercept': f"{intercept:.3f}",
                                'R-squared': f"{r_value**2:.3f}", 'p-value': f"{p_value:.3f}"
                            })
                        else:
                            regression_results_list_meta.append({
                                'Study': study, 'Slope': "N/A (Insufficient Data)", 'Intercept': "N/A (Insufficient Data)",
                                'R-squared': "N/A (Insufficient Data)", 'p-value': "N/A (Insufficient Data)"
                            })

                    fig_regression_meta.update_layout(
                        title=f'RVNA Proportion vs. Days for {regression_group_meta} Group',
                        xaxis_title='Days', yaxis_title='Proportion of RVNA >0.5IU/L', hovermode="closest"
                    )
                    st.plotly_chart(fig_regression_meta, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### Individual Study Regression Results:")
                    st.write("This table summarizes the linear regression findings for each study.")
                    if regression_results_list_meta:
                        regression_results_df_meta = pd.DataFrame(regression_results_list_meta)
                        st.dataframe(regression_results_df_meta, use_container_width=True)
                        st.markdown("""
                        **Explanation of Regression Metrics:**
                        * **Slope:** Represents the average change in RVNA Proportion for each additional day.
                        * **Intercept:** The predicted RVNA Proportion at Day 0.
                        * **R-squared:** Indicates the proportion of variance in RVNA Proportion explained by 'Days'.
                        * **p-value:** Determines the statistical significance of the slope.
                        """)
                    else:
                        st.info("No regression results to display.")

    with tabs[1]:
        st.subheader("Risk of Bias Analysis")
        st.info("This section is under development. Please provide details on how you would like to implement Risk of Bias analysis.")

    with tabs[2]:
        st.subheader("Sensitivity Analysis")
        st.info("This section is under development. Please specify the type of sensitivity analysis you'd like to perform.")

    with tabs[3]:
        st.subheader("Baujat Plot")
        st.info("This section is under development. A Baujat plot is used to visualize heterogeneity in meta-analysis.")

    with tabs[4]:
        st.subheader("Funnel Plot")
        st.info("This section is under development. A Funnel plot is used to assess publication bias in meta-analysis.")

# --- Observational Data Analysis Application Logic ---
def run_observational_analysis_app(file_input, file_name=None, is_sample_data=False):
    st.title("🧪 Hemoglobin Level Regression Analysis")
    st.markdown("""
    This application performs a multiple linear regression analysis to understand the relationship
    between 'tea in grams', 'therapy duration', and 'change in hemoglobin levels'.
    """)

    st.markdown("---")

    try:
        if is_sample_data:
            df = pd.read_excel(file_input, header=[0, 1])
        else:
            file_extension = os.path.splitext(file_name)[1].lower()
            file_input.seek(0)
            if file_extension == ".csv":
                df = pd.read_csv(file_input, header=[0, 1])
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_input, header=[0, 1])
            else:
                st.error("Unsupported file type for observational analysis. Please upload a CSV, XLSX, or XLS file.")
                st.stop()
    except Exception as e:
        st.error(f"Error reading file for observational analysis: {e}.")
        st.stop()

    new_columns = []
    for col_idx, col_tuple in enumerate(df.columns):
        if isinstance(col_tuple, tuple):
            part1 = str(col_tuple[0]).strip() if col_tuple[0] is not None else ''
            part2 = str(col_tuple[1]).strip() if col_tuple[1] is not None else ''
            if part1.startswith('Unnamed:'): part1 = ''
            if part2.startswith('Unnamed:'): part2 = ''
            if part1 and part2: new_col_name = f"{part1}_{part2}"
            elif part1: new_col_name = part1
            elif part2: new_col_name = part2
            else: new_col_name = f"Column_{col_idx}"
            new_columns.append(new_col_name)
        else:
            new_columns.append(str(col_tuple).strip())

    col_counts = Counter(new_columns)
    final_unique_columns = []
    for col_name in new_columns:
        if col_counts[col_name] > 1:
            count = 1
            original_col_name = col_name
            while f"{original_col_name}_{count}" in final_unique_columns or f"{original_col_name}_{count}" in new_columns[new_columns.index(col_name) + 1:]:
                count += 1
            final_unique_columns.append(f"{original_col_name}_{count}")
        else:
            final_unique_columns.append(col_name)

    df.columns = final_unique_columns
    st.subheader("Raw Data Preview (after header processing)")
    st.write(df.head())

    df.columns = df.columns.str.strip()

    possible_independent_var1_names = ['calculations_tea/coffee in grams', 'tea']
    possible_independent_var2_names = ['calculations_therapy length']
    possible_dependent_var_names = ['calculations_change in hb']

    independent_var1 = None
    independent_var2 = None
    dependent_var = None

    for name in possible_independent_var1_names:
        if name in df.columns:
            independent_var1 = name
            break
    if independent_var1 is None:
        st.error(f"Error: Could not find 'tea in grams' column. Looked for: {', '.join(possible_independent_var1_names)}.")
        st.stop()

    for name in possible_independent_var2_names:
        if name in df.columns:
            independent_var2 = name
            break

    if independent_var2 is None:
        st.info("Attempting to calculate 'therapy duration' from 'Hemoglobin_Start' and 'Hemoglobin_End' dates.")
        start_date_col = None
        end_date_col = None
        for col in df.columns:
            if 'Hemoglobin_Start' in col: start_date_col = col
            if 'Hemoglobin_End' in col: end_date_col = col

        if start_date_col and end_date_col:
            df[start_date_col] = pd.to_datetime(df[start_date_col], errors='coerce')
            df[end_date_col] = pd.to_datetime(df[end_date_col], errors='coerce')
            df['therapy_duration_days'] = (df[end_date_col] - df[start_date_col]).dt.days
            independent_var2 = 'therapy_duration_days'
        else:
            st.error("Error: Could not find 'calculations_therapy length' or 'Hemoglobin_Start' and/or 'Hemoglobin_End' date columns.")
            st.stop()

    for name in possible_dependent_var_names:
        if name in df.columns:
            dependent_var = name
            break

    if dependent_var is None:
        st.info("Attempting to calculate 'change in hemoglobin' from 'Hemoglobin_End' and 'Baseline'.")
        final_hemoglobin_col = None
        if 'Hemoglobin' in df.columns: final_hemoglobin_col = 'Hemoglobin'
        elif 'Hemoglobin_End' in df.columns: final_hemoglobin_col = 'Hemoglobin_End'

        if final_hemoglobin_col and 'Baseline' in df.columns:
            df['Baseline'] = pd.to_numeric(df['Baseline'], errors='coerce')
            df[final_hemoglobin_col] = pd.to_numeric(df[final_hemoglobin_col], errors='coerce')
            df['change_in_hemoglobin_calculated'] = df[final_hemoglobin_col] - df['Baseline']
            dependent_var = 'change_in_hemoglobin_calculated'
        else:
            st.error("Error: Could not find 'calculations_change in hb' or sufficient columns to calculate the change in hemoglobin.")
            st.stop()

    required_columns = [independent_var1, independent_var2, dependent_var]
    missing_columns_after_identification = [col for col in required_columns if col not in df.columns]

    if missing_columns_after_identification:
        st.error(f"Critical Error: Missing required columns: {', '.join(missing_columns_after_identification)}.")
        st.stop()
    else:
        st.success(f"Identified independent variables: '{independent_var1}', '{independent_var2}'")
        st.success(f"Identified dependent variable: '{dependent_var}'")

        st.subheader("Data Preprocessing")
        st.info("Cleaning data by converting relevant columns to numeric and dropping rows with missing values.")

        df[independent_var1] = pd.to_numeric(df[independent_var1], errors='coerce')
        df[independent_var2] = pd.to_numeric(df[independent_var2], errors='coerce')
        df[dependent_var] = pd.to_numeric(df[dependent_var], errors='coerce')

        df_cleaned = df.dropna(subset=[independent_var1, independent_var2, dependent_var])

        if df_cleaned.empty:
            st.warning("After cleaning, no valid data rows remain for analysis.")
        else:
            st.write(f"Original rows: {len(df)}, Rows after cleaning: {len(df_cleaned)}")
            st.subheader("Cleaned Data Preview")
            st.write(df_cleaned.head())

            st.subheader("Regression Analysis")
            st.write("This section presents the results of the multiple linear regression model.")
            st.markdown("---")
            formula = f"Q('{dependent_var}') ~ Q('{independent_var1}') + Q('{independent_var2}')"

            try:
                model = smf.ols(formula=formula, data=df_cleaned).fit()
                st.success("Regression model fitted successfully!")

                st.subheader("Regression Results Summary")
                st.write("Below is the full statistical summary of the regression model.")
                st.write(model.summary())

                st.subheader("Key Insights from the Analysis")
                st.markdown("""
                The regression analysis helps us understand how 'tea in grams' and 'therapy duration'
                individually and collectively influence the 'change in hemoglobin levels'.
                """)

                st.markdown("---")
                st.markdown("#### 1. R-squared Value")
                st.write(f"**R-squared:** `{model.rsquared:.4f}`")
                st.markdown("""
                The R-squared value indicates the proportion of the variance in the dependent variable
                that can be predicted from the independent variables.
                """)

                st.markdown("---")
                st.markdown("#### 2. Coefficients and P-values")
                st.markdown("""
                The coefficients tell us the average change in the dependent variable for a one-unit increase
                in the independent variable, holding other independent variables constant.
                """)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("##### Coefficients:")
                    st.write(model.params.round(4))
                with col2:
                    st.write("##### P-values:")
                    st.write(model.pvalues.round(4))

                st.markdown("---")
                st.markdown("#### 3. Interpretation of Variables:")

                st.write(f"- **Intercept:** `{model.params['Intercept']:.4f}`")
                st.markdown("""
                This is the predicted 'change in hemoglobin' when both 'tea in grams' and 'therapy duration' are zero.
                """)

                tea_coeff = model.params[f"Q('{independent_var1}')"]
                tea_pvalue = model.pvalues[f"Q('{independent_var1}')"]
                st.write(f"- **Tea in Grams (Coefficient: `{tea_coeff:.4f}`, P-value: `{tea_pvalue:.4f}`):**")
                if tea_pvalue < 0.05:
                    st.markdown(f"""
                    For every one-unit increase in 'tea in grams', the 'change in hemoglobin' is predicted to change by `{tea_coeff:.4f}` units,
                    assuming 'therapy duration' remains constant. This variable is **statistically significant**.
                    """)
                else:
                    st.markdown(f"""
                    For every one-unit increase in 'tea in grams', the 'change in hemoglobin' is predicted to change by `{tea_coeff:.4f}` units,
                    assuming 'therapy duration' remains constant. This variable is **not statistically significant**.
                    """)

                therapy_coeff = model.params[f"Q('{independent_var2}')"]
                therapy_pvalue = model.pvalues[f"Q('{independent_var2}')"]
                st.write(f"- **Therapy Duration (Coefficient: `{therapy_coeff:.4f}`, P-value: `{therapy_pvalue:.4f}`):**")
                if therapy_pvalue < 0.05:
                    st.markdown(f"""
                    For every one-unit increase in 'therapy duration', the 'change in hemoglobin' is predicted to change by `{therapy_coeff:.4f}` units,
                    assuming 'tea in grams' remains constant. This variable is **statistically significant**.
                    """)
                else:
                    st.markdown(f"""
                    For every one-unit increase in 'therapy duration', the 'change in hemoglobin' is predicted to change by `{therapy_coeff:.4f}` units,
                    assuming 'tea in grams' remains constant. This variable is **not statistically significant**.
                    """)

                st.markdown("---")
                st.subheader("Visualizations")
                st.write("These plots help visualize the relationships between variables and assess the model's assumptions.")

                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=independent_var1, y=dependent_var, data=df_cleaned, ax=ax1)
                    ax1.set_title(f'Scatter Plot: {independent_var1} vs. {dependent_var}')
                    ax1.set_xlabel(independent_var1)
                    ax1.set_ylabel(dependent_var)
                    st.pyplot(fig1)
                with col_plot2:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=independent_var2, y=dependent_var, data=df_cleaned, ax=ax2)
                    ax2.set_title(f'Scatter Plot: {independent_var2} vs. {dependent_var}')
                    ax2.set_xlabel(independent_var2)
                    ax2.set_ylabel(dependent_var)
                    st.pyplot(fig2)

                df_cleaned['residuals'] = model.resid
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=model.fittedvalues, y=df_cleaned['residuals'], ax=ax3)
                ax3.axhline(y=0, color='r', linestyle='--')
                ax3.set_title('Residuals vs. Fitted Values Plot')
                ax3.set_xlabel('Fitted Values (Predicted Hemoglobin Change)')
                ax3.set_ylabel('Residuals')
                st.pyplot(fig3)
                st.markdown("""
                The residuals plot helps to check the assumptions of linear regression.
                Ideally, residuals should be randomly scattered around zero, with no clear pattern.
                """)

                st.markdown("---")
                st.subheader("Further Considerations")
                st.markdown("""
                * **Causation vs. Correlation:** Regression analysis shows correlation, not necessarily causation.
                * **Data Quality:** The quality of the insights depends on the quality and representativeness of your input data.
                * **Model Assumptions:** Linear regression relies on several assumptions.
                * **Outliers:** Extreme values in your data can significantly impact the regression results.
                """)

            except Exception as e:
                st.error(f"An error occurred during regression analysis: {e}")

# --- Main App Logic ---
def main():
    analysis_type, uploaded_file, sample_file_path = home_page()

    if uploaded_file is not None or sample_file_path is not None:
        st.markdown("---")
        st.subheader("Raw Data Preview (from uploaded file)")
        st.write("Here's a glimpse of the data you've uploaded or selected:")
        
        try:
            if sample_file_path:
                df_initial_preview = pd.read_excel(sample_file_path)
                file_name = os.path.basename(sample_file_path)
                is_sample_data = True
            else:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                uploaded_file.seek(0)
                if file_extension == ".csv":
                    df_initial_preview = pd.read_csv(uploaded_file)
                elif file_extension in [".xlsx", ".xls"]:
                    df_initial_preview = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload a CSV, XLSX, or XLS file.")
                    st.stop()
                file_name = uploaded_file.name
                is_sample_data = False

            # Store the dataframe in session state to persist it across reruns
            if 'df_initial_preview' not in st.session_state:
                st.session_state.df_initial_preview = df_initial_preview

            with st.expander("Click to view Raw Data Preview"):
                st.dataframe(st.session_state.df_initial_preview.head())
            
        except Exception as e:
            st.error(f"Error reading the file for initial preview: {e}.")
            st.stop()

        st.markdown("---")

        if analysis_type == "Meta-Analysis":
            run_meta_analysis_app(st.session_state.df_initial_preview)
        elif analysis_type == "Observational Data Analysis":
            if sample_file_path:
                run_observational_analysis_app(sample_file_path, file_name, is_sample_data=True)
            else:
                run_observational_analysis_app(uploaded_file, file_name, is_sample_data=False)
    else:
        st.info("Please upload a CSV or Excel file or click 'Use Sample Data' to begin the analysis.")

if __name__ == "__main__":
    main()