import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, chi2, linregress # Added linregress for regression
import statsmodels.api as sm # For multiple linear regression
import statsmodels.formula.api as smf # For formula-based regression

# Streamlit app title
st.title("RVNA Proportion and Forest Plot Visualization")

# Overall software description
st.markdown(
    """
    **User-friendly statistical analysis software for physicians with minimal statistical and coding knowledge.**
    Physicians can directly upload their dataset via Excel/CSV and select which action to perform.
    Currently, this tool supports generating bar graphs and forest plots.
    """
)

st.write("Visualize the proportion of RVNA >0.5IU/L or Adverse Events for Intervention and Control groups across different studies.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

df = None # Initialize df to None

if uploaded_file is not None:
    try:
        # Determine the file type and read it.
        # For 'tea.xlsx - Prospective.csv', the header is on the second row (index 1)
        # For 'Rabies.xlsx - Sheet2.csv', the header is on the first row (index 0)
        
        # For initial loading, assume a single header for general display.
        # Specific header handling will be done within analysis types.
        if uploaded_file.name.endswith('.csv'):
            # Try reading with different delimiters if it fails
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, sep=';') # Try semicolon as delimiter
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}. Please ensure it's a valid CSV or Excel file and correctly formatted.")
        st.stop() # Stop the app if data cannot be loaded
else:
    st.info("Please upload a CSV or Excel file to proceed.")
    st.stop() # Stop the app execution if no file is uploaded

# --- Raw Data Preview (Moved to Top) ---
st.markdown("---")
st.subheader("Raw Data Preview")
st.dataframe(df)
st.markdown("---") # Add a separator after the raw data preview

# --- Data Preprocessing for Meta-Analysis (Conditional) ---
# Define the new column names for the Meta-Analysis data structure
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


# --- Function to prepare data for Forest Plot (Risk Ratio and CI) ---
def calculate_risk_ratio_and_ci(df_input, outcome_prefix):
    """
    Calculates Risk Ratio (RR) and its 95% Confidence Interval for each study.
    Applies continuity correction for zero counts.
    Returns a DataFrame with Study, Events (Intervention), Total (Intervention),
    Events (Control), Total (Control), RR, RR_Lower_CI, RR_Upper_CI.
    """
    df_calc = df_input.copy()

    # Define columns for the current outcome
    int_count_col = f'{outcome_prefix}_Intervention_Count'
    int_pop_col = f'{outcome_prefix}_Intervention_Population'
    ctrl_count_col = f'{outcome_prefix}_Control_Count'
    ctrl_pop_col = f'{outcome_prefix}_Control_Population'

    # Ensure columns exist and are numeric
    for col in [int_count_col, int_pop_col, ctrl_count_col, ctrl_pop_col]:
        if col not in df_calc.columns:
            # If a critical column is missing, return empty DataFrame
            st.error(f"Missing expected column for Forest Plot calculation: {col}")
            return pd.DataFrame()
        df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce') # Corrected: use df_calc[col]

    # Calculate proportions without dropping rows yet
    df_calc['a'] = df_calc[int_count_col] # Events in Intervention
    df_calc['n1'] = df_calc[int_pop_col] # Total in Intervention
    df_calc['c'] = df_calc[ctrl_count_col] # Events in Control
    df_calc['n0'] = df_calc[ctrl_pop_col] # Total in Control

    # Initialize RR and CI columns to NaN
    df_calc['RR'] = np.nan
    df_calc['RR_Lower_CI'] = np.nan
    df_calc['RR_Upper_CI'] = np.nan
    df_calc['log_RR'] = np.nan
    df_calc['V_log_RR'] = np.nan

    # Filter for rows where calculation is possible (non-zero populations)
    # Perform calculations only on valid rows
    valid_rows_mask = (df_calc['n1'] > 0) & (df_calc['n0'] > 0)
    df_valid = df_calc[valid_rows_mask].copy()

    if not df_valid.empty:
        # Apply continuity correction (add 0.5 to counts if any count or non-count is zero)
        df_valid['a_corr'] = df_valid['a'].apply(lambda x: x + 0.5 if x == 0 else x)
        df_valid['n1_corr'] = df_valid['n1'].apply(lambda x: x + 0.5 if x == 0 else x)
        df_valid['c_corr'] = df_valid['c'].apply(lambda x: x + 0.5 if x == 0 else x)
        df_valid['n0_corr'] = df_valid['n0'].apply(lambda x: x + 0.5 if x == 0 else x)

        # Calculate proportions with corrected counts
        df_valid['p1'] = df_valid['a_corr'] / df_valid['n1_corr']
        df_valid['p0'] = df_valid['c_corr'] / df_valid['n0_corr']

        # Calculate Risk Ratio (RR)
        df_valid['RR'] = df_valid['p1'] / df_valid['p0']
        df_valid['log_RR'] = np.log(df_valid['RR'])

        # Calculate Standard Error of log(RR)
        df_valid['SE_log_RR'] = np.sqrt(
            (1 - df_valid['p1']) / df_valid['a_corr'] +
            (1 - df_valid['p0']) / df_valid['c_corr']
        )
        df_valid['V_log_RR'] = df_valid['SE_log_RR']**2 # Variance of log(RR)

        # Calculate 95% Confidence Interval for log(RR)
        z_score = norm.ppf(0.975) # For 95% CI (1.96 approx)
        df_valid['log_RR_lower'] = df_valid['log_RR'] - z_score * df_valid['SE_log_RR']
        df_valid['log_RR_upper'] = df_valid['log_RR'] + z_score * df_valid['SE_log_RR']

        # Convert back to RR scale
        df_valid['RR_Lower_CI'] = np.exp(df_valid['log_RR_lower'])
        df_valid['RR_Upper_CI'] = np.exp(df_valid['log_RR_upper'])

        # Update the original df_calc with calculated values
        df_calc.loc[valid_rows_mask, ['RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']] = \
            df_valid[['RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']]

    # Select relevant columns for plotting and table display, including original counts/populations
    return df_calc[['Study', 'a', 'n1', 'c', 'n0', 'RR', 'RR_Lower_CI', 'RR_Upper_CI', 'log_RR', 'V_log_RR']]


def perform_meta_analysis(forest_df):
    """
    Performs fixed-effect (Inverse Variance) and random-effects (DerSimonian-Laird) meta-analysis.
    Returns a dictionary with overall results and heterogeneity stats.
    """
    # Filter out rows where RR calculation was not possible (NaNs in log_RR or V_log_RR)
    df_for_meta = forest_df.dropna(subset=['log_RR', 'V_log_RR']).copy()

    if df_for_meta.empty:
        # If no valid studies for meta-analysis, return None
        return None

    # Fixed-Effect (Inverse Variance)
    df_for_meta['w_fixed'] = 1 / df_for_meta['V_log_RR']
    sum_w_fixed = df_for_meta['w_fixed'].sum()

    fixed_weights = pd.Series(np.nan, index=df_for_meta.index)
    log_RR_fixed_overall = np.nan
    SE_log_RR_fixed_overall = np.nan
    RR_fixed_overall = np.nan
    RR_fixed_lower_ci = np.nan
    RR_fixed_upper_ci = np.nan

    if sum_w_fixed > 0: # Only calculate if sum of weights is positive
        fixed_weights = df_for_meta['w_fixed'] / sum_w_fixed * 100
        log_RR_fixed_overall = (df_for_meta['w_fixed'] * df_for_meta['log_RR']).sum() / sum_w_fixed
        SE_log_RR_fixed_overall = np.sqrt(1 / sum_w_fixed)
        RR_fixed_overall = np.exp(log_RR_fixed_overall)
        RR_fixed_lower_ci = np.exp(log_RR_fixed_overall - norm.ppf(0.975) * SE_log_RR_fixed_overall)
        RR_fixed_upper_ci = np.exp(log_RR_fixed_overall + norm.ppf(0.975) * SE_log_RR_fixed_overall)


    # Heterogeneity (Cochran's Q)
    # Q calculation depends on log_RR_fixed_overall, which might be NaN if sum_w_fixed was 0
    Q = np.nan
    if pd.notna(log_RR_fixed_overall):
        Q = (df_for_meta['w_fixed'] * (df_for_meta['log_RR'] - log_RR_fixed_overall)**2).sum()

    df_heterogeneity = len(df_for_meta) - 1
    if df_heterogeneity < 0: df_heterogeneity = 0 # Handle single study case

    p_heterogeneity = 1.0 # Default if df is 0 or Q is very small or NaN
    if df_heterogeneity > 0 and pd.notna(Q):
        p_heterogeneity = 1 - chi2.cdf(Q, df_heterogeneity)

    I2 = 0.0 # Default
    if pd.notna(Q) and Q > 0 and df_heterogeneity > 0:
        I2 = max(0, ((Q - df_heterogeneity) / Q) * 100)

    tau_squared = 0.0 # Default
    C = sum_w_fixed - (df_for_meta['w_fixed']**2).sum() / sum_w_fixed if sum_w_fixed > 0 else 0
    if C > 0 and pd.notna(Q) and df_heterogeneity > 0:
        tau_squared = max(0, (Q - df_heterogeneity) / C)


    # Random-Effects (DerSimonian-Laird)
    # Recalculate weights for random effect including tau_squared
    df_for_meta['w_random'] = 1 / (df_for_meta['V_log_RR'] + tau_squared)
    sum_w_random = df_for_meta['w_random'].sum()

    random_weights = pd.Series(np.nan, index=df_for_meta.index)
    log_RR_random_overall = np.nan
    SE_log_RR_random_overall = np.nan
    RR_random_overall = np.nan
    RR_random_lower_ci = np.nan
    RR_random_upper_ci = np.nan

    if sum_w_random > 0: # Only calculate if sum of weights is positive
        random_weights = df_for_meta['w_random'] / sum_w_random * 100
        log_RR_random_overall = (df_for_meta['w_random'] * df_for_meta['log_RR']).sum() / sum_w_random
        SE_log_RR_random_overall = np.sqrt(1 / sum_w_random)
        RR_random_overall = np.exp(log_RR_random_overall)
        RR_random_lower_ci = np.exp(log_RR_random_overall - norm.ppf(0.975) * SE_log_RR_random_overall)
        RR_random_upper_ci = np.exp(log_RR_random_overall + norm.ppf(0.975) * SE_log_RR_random_overall)

    # Total events and population for overall summary (from the studies used in meta-analysis)
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
            'fixed': fixed_weights, # These are Series, will be handled in display
            'random': random_weights # These are Series, will be handled in display
        }
    }


# --- Top-level Analysis Category Selection using tabs ---
tab_titles = ["Statistical Analysis", "Risk of Bias", "Sensitivity Analysis", "Baujat Plot", "Funnel Plot"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Statistical Analysis Tab
    # Create a copy of the original DataFrame for meta-analysis specific processing
    df_meta = df.copy()

    # Check if the number of columns in the uploaded DataFrame matches the expected number for Meta-Analysis
    expected_num_cols_meta = len(meta_analysis_column_names_list)
    if len(df_meta.columns) != expected_num_cols_meta:
        st.warning(f"The uploaded file does not seem to have the expected number of columns for Meta-Analysis. "
                   f"Expected {expected_num_cols_meta} columns, but found {len(df_meta.columns)}. "
                   "Please ensure your file matches the required data structure for Meta-Analysis "
                   "(e.g., 'Rabies.xlsx - Sheet2.csv').")
        st.stop() # Stop if column structure is unexpected for meta-analysis

    # Assign the new column names directly to the DataFrame for meta-analysis
    df_meta.columns = meta_analysis_column_names_list

    # Convert count and population columns to numeric, coercing errors to NaN
    numeric_cols_meta = [col for col in df_meta.columns if 'Count' in col or 'Population' in col]
    for col in numeric_cols_meta:
        df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce')

    # Calculate proportions for all added categories for meta-analysis
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


    # --- Streamlit UI for Plot Type Selection within Statistical Analysis ---
    plot_type_meta = st.radio(
        "Choose Plot Type for Statistical Analysis:",
        ("Bar Graph", "Forest Plot", "RVNA Proportion vs. Days Regression")
    )

    # --- Streamlit UI for Data Selection (common for Statistical Analysis plots, except regression) ---
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
            )
        )

    # --- Plotting Logic for Statistical Analysis ---
    if plot_type_meta == "Bar Graph":
        df_plot = pd.DataFrame() # Initialize to avoid UnboundLocalError
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


        # Melt the DataFrame for Plotly Express to create grouped bars
        df_melted = df_plot.melt(id_vars=['Study'], var_name='Group', value_name='Proportion')

        # Remove NaN values from melted DataFrame before plotting, as they can cause issues
        # This ensures only valid proportion values are plotted for the selected graph.
        df_melted.dropna(subset=['Proportion'], inplace=True)

        # Create the bar chart using Plotly Express
        if not df_melted.empty: # Only plot if there's data after dropping NaNs
            fig = px.bar(
                df_melted,
                x='Study',
                y='Proportion',
                color='Group',
                barmode='group', # This creates grouped bars for each study
                title=title,
                labels={'Proportion': y_axis_label, 'Study': 'Study'},
                color_discrete_map={'Intervention': '#4CAF50', 'Control': '#FFC107'} # Consistent colors
            )

            # Customize layout for better readability
            fig.update_layout(
                xaxis_title="Study",
                yaxis_title=y_axis_label,
                xaxis_tickangle=-45, # Rotate x-axis labels for readability
                yaxis_range=[0, 1.1], # Ensure y-axis goes from 0 to 1 (proportion) and slightly above
                legend_title_text='Group'
            )

            # Add text labels on top of the bars for exact values
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available to plot for the selected visualization option after processing. "
                       "This might be due to missing or non-numeric values in the relevant columns for this selection.")

    elif plot_type_meta == "Forest Plot":
        outcome_prefix_map = {
            "RVNA >0.5IU/L at 0 days": "RVNA_0d",
            "RVNA >0.5IU/L at 14 days": "RVNA_14d",
            "RVNA >0.5IU/L at 42 days": "RVNA_42d",
            "RVNA >0.5IU/L at 82 days": "RVNA_82d",
            "All Adverse Events": "All_AE",
            "Local Adverse Events": "Local_AE"
        }
        selected_outcome_prefix = outcome_prefix_map.get(graph_option_meta)

        if selected_outcome_prefix:
            forest_df = calculate_risk_ratio_and_ci(df_meta, selected_outcome_prefix) # Use df_meta

            if not forest_df.empty:
                # Perform meta-analysis
                meta_results = perform_meta_analysis(forest_df)

                # --- Forest Plot (main plot) ---
                fig_forest = go.Figure()

                # Individual study points with error bars
                # Filter forest_df for plotting to only include studies with valid RR
                forest_df_plot = forest_df.dropna(subset=['RR', 'RR_Lower_CI', 'RR_Upper_CI']).copy()

                fig_forest.add_trace(go.Scatter(
                    x=forest_df_plot['RR'],
                    y=forest_df_plot['Study'],
                    mode='markers',
                    marker=dict(color='blue', size=10, symbol='square'), # Square markers for individual studies
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=forest_df_plot['RR_Upper_CI'] - forest_df_plot['RR'],
                        arrayminus=forest_df_plot['RR'] - forest_df_plot['RR_Lower_CI'], # Corrected column name
                        color='blue',
                        thickness=1.5,
                        width=5
                    ),
                    name='Individual Studies'
                ))

                if meta_results:
                    # Fixed effect diamond
                    if pd.notna(meta_results['fixed_effect']['RR']):
                        fig_forest.add_trace(go.Scatter(
                            x=[meta_results['fixed_effect']['RR']],
                            y=["Overall (Fixed-effect)"], # Use a distinct label for the overall effect
                            mode='markers',
                            marker=dict(
                                symbol='diamond',
                                size=20, # Larger diamond
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

                    # Random effect diamond
                    if pd.notna(meta_results['random_effects']['RR']):
                        fig_forest.add_trace(go.Scatter(
                            x=[meta_results['random_effects']['RR']],
                            y=["Overall (Random-effects)"], # Use a distinct label for the overall effect
                            mode='markers',
                            marker=dict(
                                symbol='diamond',
                                size=20, # Larger diamond
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

                # Add vertical line at RR = 1 (no effect)
                fig_forest.add_shape(
                    type="line",
                    x0=1, x1=1, y0=0, y1=1,
                    xref='x', yref='paper', # Target the plot's x-axis and span full paper height
                    line=dict(color="red", width=1, dash="dash"),
                )

                # Add annotation for "No Effect"
                fig_forest.add_annotation(
                    x=1, y=1.02, # Position it slightly above the top of the plot
                    xref="x", yref="paper", # Target the plot's x-axis and paper coordinates for y
                    text="No Effect",
                    showarrow=False,
                    font=dict(color="red", size=10),
                    xanchor="left" # Anchor text to the left of x=1
                )


                # Customize layout for the forest plot
                fig_forest.update_layout(
                    title=f'Forest Plot: Risk Ratio for {graph_option_meta}',
                    xaxis_title='Risk Ratio (Intervention vs. Control)',
                    yaxis_title='Study',
                    yaxis_autorange="reversed", # Studies from top to bottom
                    hovermode="y unified",
                    height=600, # Adjust height as needed
                    showlegend=False # Hide individual trace legends
                )

                # Set y-axis categories including overall labels
                y_axis_categories = forest_df_plot['Study'].tolist() # Only studies that are plotted
                if meta_results and pd.notna(meta_results['fixed_effect']['RR']):
                    y_axis_categories.append("Overall (Fixed-effect)")
                if meta_results and pd.notna(meta_results['random_effects']['RR']):
                    y_axis_categories.append("Overall (Random-effects)")
                fig_forest.update_yaxes(categoryorder='array', categoryarray=y_axis_categories)


                # Set x-axis range dynamically based on data, but ensure 1 is visible
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
                    min_rr = min(min_rr_data * 0.9, 0.9) # Ensure 0.9 is visible
                    max_rr = max(max_rr_data * 1.1, 1.1) # Ensure 1.1 is visible
                else:
                    min_rr, max_rr = 0.5, 1.5 # Default range if no valid RR data

                fig_forest.update_xaxes(range=[min_rr, max_rr])


                # Display the Forest Plot
                st.plotly_chart(fig_forest, use_container_width=True)

                # --- Table (below the plot) ---
                st.markdown("---")
                st.subheader("Detailed Study Data and Meta-Analysis Results")

                if meta_results:
                    # Create a DataFrame for the table display
                    # Use the original forest_df to include all studies, even if not plotted
                    table_data = forest_df[['Study', 'a', 'n1', 'c', 'n0', 'RR', 'RR_Lower_CI', 'RR_Upper_CI']].copy()

                    # Add formatted RR and CI
                    table_data['RR (95% CI)'] = table_data.apply(
                        lambda row: f"{row['RR']:.2f} [{row['RR_Lower_CI']:.2f}, {row['RR_Upper_CI']:.2f}]"
                                    if pd.notna(row['RR']) and pd.notna(row['RR_Lower_CI']) and pd.notna(row['RR_Upper_CI'])
                                    else "N/A", axis=1
                    )

                    # Add weights for individual studies, mapping back to original forest_df index
                    # Ensure weights are aligned with the original forest_df for display
                    table_data['Weight (Fixed)'] = ""
                    table_data['Weight (Random)'] = ""

                    # Populate weights only for studies that were included in meta-analysis
                    if not meta_results['study_weights']['fixed'].empty:
                        # Iterate through the studies that were part of the meta-analysis calculation
                        for meta_idx, study_row in forest_df_plot.iterrows():
                            original_df_idx = forest_df[forest_df['Study'] == study_row['Study']].index
                            if not original_df_idx.empty:
                                weight = meta_results['study_weights']['fixed'].get(meta_idx)
                                if pd.notna(weight):
                                    table_data.loc[original_df_idx[0], 'Weight (Fixed)'] = f"{weight:.1f}%"
                                else:
                                    table_data.loc[original_df_idx[0], 'Weight (Fixed)'] = "N/A"
                
                if not meta_results['study_weights']['random'].empty:
                    # Iterate through the studies that were part of the meta-analysis calculation
                    for meta_idx, study_row in forest_df_plot.iterrows():
                        original_df_idx = forest_df[forest_df['Study'] == study_row['Study']].index
                        if not original_df_idx.empty:
                            weight = meta_results['study_weights']['random'].get(meta_idx)
                            if pd.notna(weight):
                                table_data.loc[original_df_idx[0], 'Weight (Random)'] = f"{weight:.1f}%"
                            else:
                                table_data.loc[original_df_idx[0], 'Weight (Random)'] = "N/A"


                    # Prepare overall rows
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
                        'Weight (Fixed)': "100.0%",
                        'Weight (Random)': ""
                    }
                    overall_random_row = {
                        'Study': "Overall (Random-effects)",
                        'a': "", 'n1': "", 'c': "", 'n0': "", # N/A for these columns in overall random
                        'RR': meta_results['random_effects']['RR'].round(2) if pd.notna(meta_results['random_effects']['RR']) else np.nan,
                        'RR_Lower_CI': meta_results['random_effects']['Lower_CI'].round(2) if pd.notna(meta_results['random_effects']['Lower_CI']) else np.nan,
                        'RR_Upper_CI': meta_results['random_effects']['Upper_CI'].round(2) if pd.notna(meta_results['random_effects']['Upper_CI']) else np.nan,
                        'RR (95% CI)': f"{meta_results['random_effects']['RR']:.2f} [{meta_results['random_effects']['Lower_CI']:.2f}, {meta_results['random_effects']['Upper_CI']:.2f}]" if pd.notna(meta_results['random_effects']['RR']) else "N/A",
                        'Weight (Fixed)': "",
                        'Weight (Random)': "100.0%"
                    }

                    # Append overall rows to the table data
                    table_data = pd.concat([table_data, pd.DataFrame([overall_fixed_row, overall_random_row])], ignore_index=True)

                    # Rename columns for display
                    table_data.rename(columns={
                        'a': 'Events (Exp)',
                        'n1': 'Total (Exp)',
                        'c': 'Events (Ctrl)',
                        'n0': 'Total (Ctrl)'
                    }, inplace=True)

                    # Select and reorder columns for final display
                    display_cols = [
                        'Study', 'Events (Exp)', 'Total (Exp)',
                        'Events (Ctrl)', 'Total (Ctrl)',
                        'RR (95% CI)', 'Weight (Fixed)', 'Weight (Random)'
                    ]
                    st.dataframe(table_data[display_cols].fillna(''), use_container_width=True) # Fillna for empty strings

                    # Heterogeneity text below the table
                    st.markdown(f"**Heterogeneity:** $I^2 = {meta_results['heterogeneity']['I2']:.1f}\\%$, "
                                f"$\\tau^2 = {meta_results['heterogeneity']['tau_squared']:.2f}$, "
                                f"p-value = {meta_results['heterogeneity']['p_value']:.3f}")

                else:
                    st.warning(f"Could not perform meta-analysis for '{graph_option_meta}'. "
                               "Table details could not be generated.")
            else:
                st.warning(f"Could not generate Forest Plot for '{graph_option_meta}'. "
                           "This might be due to missing or invalid data for Risk Ratio calculation.")
        else:
            st.error("Invalid graph option selected for Forest Plot.")

    elif plot_type_meta == "RVNA Proportion vs. Days Regression":
        st.subheader("Regression Analysis: RVNA Proportion vs. Days (Per Study)")

        # Select group for regression
        regression_group_meta = st.selectbox(
            "Select group for RVNA Proportion vs. Days Regression:",
            ("Intervention", "Control")
        )

        # Prepare data for regression
        # Identify the correct proportion columns based on the selected group
        
        expected_proportion_cols_for_group_meta = [
            f'{regression_group_meta}_Proportion_0d',
            f'{regression_group_meta}_Proportion_14d',
            f'{regression_group_meta}_Proportion_42d',
            f'{regression_group_meta}_Proportion_82d'
        ]

        # Filter for columns that actually exist in the DataFrame
        actual_proportion_cols_meta = [col for col in expected_proportion_cols_for_group_meta if col in df_meta.columns]

        if not actual_proportion_cols_meta:
            st.warning(f"No RVNA proportion data found for {regression_group_meta} to perform regression analysis. "
                       "Please ensure your uploaded data contains RVNA proportion columns for the selected group.")
        else:
            # Create a temporary DataFrame for melting
            df_melt_regression_meta = df_meta[['Study'] + actual_proportion_cols_meta].copy()
            
            # Melt the DataFrame
            df_long_regression_meta = df_melt_regression_meta.melt(
                id_vars=['Study'],
                value_vars=actual_proportion_cols_meta,
                var_name='Proportion_Type',
                value_name='Proportion'
            )
            
            # Create a mapping for days based on the actual column names
            day_mapping_meta = {
                f'{regression_group_meta}_Proportion_0d': 0,
                f'{regression_group_meta}_Proportion_14d': 14,
                f'{regression_group_meta}_Proportion_42d': 42,
                f'{regression_group_meta}_Proportion_82d': 82,
            }

            # Map 'Proportion_Type' to 'Days'
            df_long_regression_meta['Days'] = df_long_regression_meta['Proportion_Type'].map(day_mapping_meta)
            
            # Drop rows where 'Proportion' or 'Days' is NaN (e.g., if original data was missing)
            df_long_regression_meta.dropna(subset=['Proportion', 'Days'], inplace=True)

            if df_long_regression_meta.empty:
                st.warning(f"No valid data points available for regression analysis for {regression_group_meta} after processing.")
            else:
                fig_regression_meta = go.Figure()
                
                regression_results_list_meta = [] # List to store results for table
                
                # Plot individual study data points and regression lines
                for study in df_long_regression_meta['Study'].unique():
                    study_data = df_long_regression_meta[df_long_regression_meta['Study'] == study]
                    
                    # Add scatter points for the study
                    fig_regression_meta.add_trace(go.Scatter(
                        x=study_data['Days'],
                        y=study_data['Proportion'],
                        mode='markers',
                        name=f'{study} Data',
                        marker=dict(size=8)
                    ))

                    # Ensure there are enough unique data points for regression (at least 2)
                    if len(study_data['Days'].unique()) >= 2:
                        x = study_data['Days']
                        y = study_data['Proportion']
                        
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        
                        # Add regression line for the study
                        fig_regression_meta.add_trace(go.Scatter(
                            x=x,
                            y=slope * x + intercept,
                            mode='lines',
                            name=f'{study} Trend',
                            line=dict(dash='dash')
                        ))
                        regression_results_list_meta.append({
                            'Study': study,
                            'Slope': f"{slope:.3f}",
                            'Intercept': f"{intercept:.3f}",
                            'R-squared': f"{r_value**2:.3f}",
                            'p-value': f"{p_value:.3f}"
                        })
                    else:
                        regression_results_list_meta.append({
                            'Study': study,
                            'Slope': "N/A (Insufficient Data)",
                            'Intercept': "N/A (Insufficient Data)",
                            'R-squared': "N/A (Insufficient Data)",
                            'p-value': "N/A (Insufficient Data)"
                        })

            fig_regression_meta.update_layout(
                title=f'RVNA Proportion vs. Days for {regression_group_meta} Group',
                xaxis_title='Days',
                yaxis_title='Proportion of RVNA >0.5IU/L',
                hovermode="closest"
            )
            st.plotly_chart(fig_regression_meta, use_container_width=True)
            
            st.markdown("### Individual Study Regression Results:")
            if regression_results_list_meta:
                regression_results_df_meta = pd.DataFrame(regression_results_list_meta)
                st.dataframe(regression_results_df_meta, use_container_width=True)

                st.markdown("""
                **Explanation of Regression Metrics:**
                * **Slope:** Represents the average change in RVNA Proportion for each additional day. A positive slope indicates an increasing trend, while a negative slope indicates a decreasing trend.
                * **Intercept:** The predicted RVNA Proportion at Day 0 (the starting point of the observation period).
                * **R-squared:** Indicates the proportion of the variance in RVNA Proportion that can be predicted from the 'Days' variable. A value closer to 1 means the model explains a large portion of the variance, suggesting a good fit.
                * **p-value:** Helps determine the statistical significance of the slope. A p-value less than a chosen significance level (e.g., 0.05) suggests that the observed trend is unlikely to have occurred by random chance.
                """)
            else:
                st.info("No regression results to display. This might be due to no studies having sufficient data for regression.")

with tabs[1]: # Risk of Bias Tab
    st.subheader("Risk of Bias Analysis")
    st.info("This section is under development. Please provide details on how you would like to implement Risk of Bias analysis (e.g., Cochrane RoB 2.0, traffic light plots, etc.).")

with tabs[2]: # Sensitivity Analysis Tab
    st.subheader("Sensitivity Analysis")
    st.info("This section is under development. Please specify the type of sensitivity analysis you'd like to perform (e.g., leave-one-out, subgroup analysis, impact of missing data).")

with tabs[3]: # Baujat Plot Tab
    st.subheader("Baujat Plot")
    st.info("This section is under development. A Baujat plot is used to visualize heterogeneity in meta-analysis. It requires specific calculations based on study-level contributions to heterogeneity.")

with tabs[4]: # Funnel Plot Tab
    st.subheader("Funnel Plot")
    st.info("This section is under development. A Funnel plot is used to assess publication bias in meta-analysis. It typically plots study effect size against a measure of study precision.")
