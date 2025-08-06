import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import beta
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ==============================================================================
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(layout="wide", page_title="An Interactive Guide to Assay Transfer Statistics", page_icon="📈")

st.markdown("""
<style>
    .main .block-container { padding: 1.5rem 3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; }
    .stTabs [aria-selected="true"] { background-color: #e0e0e0; font-weight: bold; }
    [data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #CCCCCC; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPER & PLOTTING FUNCTIONS
# ==============================================================================
# All plotting and helper functions are defined here.

@st.cache_data
def create_conceptual_map_plotly():
    nodes = { 'DS': ('Data Science', 0, 3.5), 'BS': ('Biostatistics', 0, 2.5), 'ST': ('Statistics', 0, 1.5), 'IE': ('Industrial Engineering', 0, 0.5), 'SI': ('Statistical Inference', 1, 2.5), 'SPC': ('SPC', 1, 0.5), 'CC': ('Control Charts', 2, 0), 'PC': ('Process Capability', 2, 1), 'WR': ('Westgard Rules', 2, 2), 'NR': ('Nelson Rules', 2, 3), 'HT': ('Hypothesis Testing', 2, 4), 'CI': ('Confidence Intervals', 2, 5), 'BAY': ('Bayesian Statistics', 2, 6), 'SWH': ('Shewhart Charts', 3, -0.5), 'EWM': ('EWMA', 3, 0), 'CSM': ('CUSUM', 3, 0.5), 'MQA': ('Manufacturing QA', 3, 1.5), 'CL': ('Clinical Labs', 3, 2.5), 'TAV': ('T-tests / ANOVA', 3, 3.5), 'ZME': ('Z-score / Margin of Error', 3, 4.5), 'WS': ('Wilson Score', 3, 5.5), 'PP': ('Posterior Probabilities', 3, 6.5), 'PE': ('Proportion Estimates', 4, 6.0) }
    edges = [('IE', 'SPC'), ('ST', 'SPC'), ('ST', 'SI'), ('BS', 'SI'), ('DS', 'SI'), ('SPC', 'CC'), ('SPC', 'PC'), ('SI', 'HT'), ('SI', 'CI'), ('SI', 'BAY'), ('SI', 'WR'), ('SI', 'NR'), ('CC', 'SWH'), ('CC', 'EWM'), ('CC', 'CSM'), ('PC', 'MQA'), ('WR', 'CL'), ('NR', 'MQA'), ('HT', 'TAV'), ('CI', 'ZME'), ('CI', 'WS'), ('BAY', 'PP'), ('WS', 'PE')]
    fig = go.Figure()
    for start, end in edges: fig.add_trace(go.Scatter(x=[nodes[start][1], nodes[end][1]], y=[nodes[start][2], nodes[end][2]], mode='lines', line=dict(color='grey', width=1), hoverinfo='none'))
    node_x = [v[1] for v in nodes.values()]; node_y = [v[2] for v in nodes.values()]; node_text = [v[0] for v in nodes.values()]; colors = ["#e0f2f1"]*4 + ["#b2dfdb"]*2 + ["#80cbc4"]*8 + ["#4db6ac"]*10
    fig.add_trace(go.Scatter(x=node_x, y=node_y, text=node_text, mode='markers+text', textposition="middle center", marker=dict(size=45, color=colors, line=dict(width=2, color='black')), textfont=dict(size=10, color='black'), hoverinfo='text'))
    fig.update_layout(title_text='Hierarchical Map of Statistical Concepts', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 7.5]), height=700, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='#FFFFFF', paper_bgcolor='#f0f2f6')
    return fig

def wilson_score_interval(p_hat, n, z=1.96):
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n; term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2))); return (term1 - term2) / denom, (term1 + term2) / denom

# ... (All 15 plotting functions are here, fully implemented and enhanced for quality)
def plot_gage_rr():
    # --- Data Generation ---
    np.random.seed(10)
    n_operators, n_samples, n_replicates = 3, 10, 3
    operators = ['Alice', 'Bob', 'Charlie']
    sample_means = np.linspace(90, 110, n_samples)
    operator_bias = {'Alice': 0, 'Bob': -0.5, 'Charlie': 0.8}
    data = []
    for op_idx, operator in enumerate(operators):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[operator], 1.5, n_replicates)
            for m_idx, m in enumerate(measurements):
                data.append([operator, f'Part_{sample_idx+1}', m, m_idx + 1])
    df = pd.DataFrame(data, columns=['Operator', 'Part', 'Measurement', 'Replicate'])
    
    # --- ANOVA Calculation ---
    model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    ms_operator = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
    ms_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
    ms_interaction = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
    ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    
    var_repeatability = ms_error
    var_reproducibility = ((ms_operator - ms_interaction) / (n_samples * n_replicates)) + ((ms_interaction - ms_error) / n_replicates)
    var_part = (ms_part - ms_interaction) / (n_operators * n_replicates)
    
    variances = {k: max(0, v) for k, v in locals().items() if 'var_' in k}
    var_rr = variances['var_repeatability'] + variances['var_reproducibility']
    var_total = var_rr + variances['var_part']
    pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0
    pct_part = (variances['var_part'] / var_total) * 100 if var_total > 0 else 0
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.5, 0.5],
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        subplot_titles=("<b>Variation by Part & Operator</b>", "<b>Overall Variation by Operator</b>", "<b>Variance Contribution</b>")
    )

    # --- Plot 1: Variation by Part & Operator (Large Plot) ---
    fig_box = px.box(df, x='Part', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_box.data:
        trace.update(hoverinfo='none', hovertemplate=None) # Hide default hover for box
        fig.add_trace(trace, row=1, col=1)
    
    # Add mean lines for each operator within each part
    for operator in operators:
        operator_df = df[df['Operator'] == operator]
        part_means = operator_df.groupby('Part')['Measurement'].mean()
        fig.add_trace(go.Scatter(
            x=part_means.index, y=part_means.values, mode='lines', 
            line=dict(width=2), name=f'{operator} Mean',
            hoverinfo='none', hovertemplate=None,
            marker_color=fig_box.data[operators.index(operator)].marker.color
        ), row=1, col=1)

    # --- Plot 2: Overall Variation by Operator ---
    fig_op_box = px.box(df, x='Operator', y='Measurement', color='Operator', color_discrete_sequence=px.colors.qualitative.Plotly)
    for trace in fig_op_box.data:
        fig.add_trace(trace, row=1, col=2)
        
    # --- Plot 3: Variance Contribution ---
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part Variation'], y=[pct_rr, pct_part], marker_color=['salmon', 'skyblue'], text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=2, col=2)
    fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=2, col=2)
    
    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Gage R&R Study: A Multi-View Dashboard</b>',
        height=800,
        showlegend=False,
        bargap=0.1,
        boxmode='group'
    )
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    return fig, pct_rr, pct_part

def plot_linearity():
    # --- Data Generation ---
    np.random.seed(42)
    nominal = np.array([10, 25, 50, 100, 150, 200, 250])
    # Introduce slight non-linearity and error
    measured = nominal + np.random.normal(0, nominal * 0.02 + 1) - (nominal / 150)**3
    
    # --- Calculations ---
    X = sm.add_constant(nominal)
    model = sm.OLS(measured, X).fit()
    b, m = model.params
    residuals = model.resid
    recovery = (measured / nominal) * 100
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("<b>Linearity Plot</b>", "<b>Residual Plot</b>", "<b>Recovery Plot</b>"),
        vertical_spacing=0.2
    )

    # --- Plot 1: Linearity Plot ---
    fig.add_trace(go.Scatter(
        x=nominal, y=measured, mode='markers', name='Measured Values',
        marker=dict(size=10, color='blue'),
        hovertemplate="Nominal: %{x}<br>Measured: %{y:.2f}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line',
        line=dict(color='red')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity',
        line=dict(dash='dash', color='black')
    ), row=1, col=1)

    # --- Plot 2: Residual Plot ---
    fig.add_trace(go.Scatter(
        x=nominal, y=residuals, mode='markers', name='Residuals',
        marker=dict(size=10, color='green'),
        hovertemplate="Nominal: %{x}<br>Residual: %{y:.2f}<extra></extra>"
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # --- Plot 3: Recovery Plot ---
    fig.add_trace(go.Scatter(
        x=nominal, y=recovery, mode='lines+markers', name='Recovery',
        line=dict(color='purple'), marker=dict(size=10),
        hovertemplate="Nominal: %{x}<br>Recovery: %{y:.1f}%<extra></extra>"
    ), row=2, col=1)
    # Add acceptance limits for recovery
    fig.add_hrect(y0=80, y1=120, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=120, line_dash="dot", line_color="red", row=2, col=1)

    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Assay Linearity and Range Verification Dashboard</b>',
        height=800,
        showlegend=False
    )
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Measured Concentration", row=1, col=1)
    fig.update_xaxes(title_text="Nominal Concentration", row=1, col=2)
    fig.update_yaxes(title_text="Residual (Error)", row=1, col=2)
    fig.update_xaxes(title_text="Nominal Concentration", row=2, col=1)
    fig.update_yaxes(title_text="% Recovery", range=[min(75, recovery.min()-5), max(125, recovery.max()+5)], row=2, col=1)
    
    return fig, model

def plot_lod_loq():
    # --- Data Generation ---
    np.random.seed(3)
    # Data for the distribution plot
    blanks_dist = np.random.normal(0.05, 0.01, 20)
    low_conc_dist = np.random.normal(0.20, 0.02, 20)
    df_dist = pd.concat([
        pd.DataFrame({'Signal': blanks_dist, 'Sample Type': 'Blank'}),
        pd.DataFrame({'Signal': low_conc_dist, 'Sample Type': 'Low Concentration'})
    ])
    
    # Data for the calibration curve method
    concentrations = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 5, 5, 5, 10, 10, 10])
    signals = 0.05 + 0.02 * concentrations + np.random.normal(0, 0.01, len(concentrations))
    df_cal = pd.DataFrame({'Concentration': concentrations, 'Signal': signals})
    
    # --- Calculations (Calibration Curve Method) ---
    X = sm.add_constant(df_cal['Concentration'])
    model = sm.OLS(df_cal['Signal'], X).fit()
    slope = model.params['Concentration']
    residual_std_err = np.sqrt(model.mse_resid)
    
    LOD = (3.3 * residual_std_err) / slope
    LOQ = (10 * residual_std_err) / slope
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("<b>Signal Distribution at Low End</b>", "<b>Low-Level Calibration Curve</b>"),
        vertical_spacing=0.2
    )
    
    # --- Plot 1: Signal Distribution ---
    fig_violin = px.violin(
        df_dist, x='Sample Type', y='Signal', color='Sample Type',
        box=True, points="all",
        color_discrete_map={'Blank': 'skyblue', 'Low Concentration': 'lightgreen'}
    )
    for trace in fig_violin.data:
        fig.add_trace(trace, row=1, col=1)

    # --- Plot 2: Low-Level Calibration Curve ---
    fig.add_trace(go.Scatter(
        x=df_cal['Concentration'], y=df_cal['Signal'], mode='markers',
        name='Calibration Points', marker=dict(color='darkblue', size=8)
    ), row=2, col=1)
    
    x_range = np.linspace(0, df_cal['Concentration'].max(), 100)
    y_range = model.predict(sm.add_constant(x_range))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range, mode='lines',
        name='Regression Line', line=dict(color='red', dash='dash')
    ), row=2, col=1)

    # Add LOD/LOQ annotations to the calibration curve
    fig.add_vline(x=LOD, line_dash="dot", line_color="orange", row=2, col=1,
                  annotation_text=f"<b>LOD = {LOD:.2f} ng/mL</b>", annotation_position="top")
    fig.add_vline(x=LOQ, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text=f"<b>LOQ = {LOQ:.2f} ng/mL</b>", annotation_position="top")

    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Assay Sensitivity Analysis: LOD & LOQ</b>',
        height=800,
        showlegend=False
    )
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=1, col=1)
    fig.update_xaxes(title_text="Sample Type", row=1, col=1)
    fig.update_yaxes(title_text="Assay Signal (e.g., Absorbance)", row=2, col=1)
    fig.update_xaxes(title_text="Concentration (ng/mL)", row=2, col=1)
    
    return fig, LOD, LOQ
    
def plot_method_comparison():
    # --- Data Generation ---
    np.random.seed(42)
    # R&D method is the 'true' reference
    x = np.linspace(20, 150, 50)
    # QC method has a small proportional and constant bias
    y = 0.98 * x + 1.5 + np.random.normal(0, 2.5, 50)
    
    # --- Calculations ---
    # Deming Regression
    delta = np.var(y, ddof=1) / np.var(x, ddof=1)
    x_mean, y_mean = np.mean(x), np.mean(y)
    Sxx = np.sum((x - x_mean)**2); Sxy = np.sum((x - x_mean)*(y - y_mean))
    beta1_deming = (np.sum((y-y_mean)**2) - delta*Sxx + np.sqrt((np.sum((y-y_mean)**2) - delta*Sxx)**2 + 4*delta*Sxy**2)) / (2*Sxy)
    beta0_deming = y_mean - beta1_deming*x_mean
    
    # Bland-Altman
    avg = (x + y) / 2
    diff = y - x
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # % Bias
    percent_bias = (diff / x) * 100
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("<b>Deming Regression</b>", "<b>Bland-Altman Agreement Plot</b>", "<b>Percent Bias vs. Concentration</b>"),
        vertical_spacing=0.2
    )

    # --- Plot 1: Deming Regression ---
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Sample Results', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=beta0_deming + beta1_deming*x, mode='lines', name='Deming Fit', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 160], y=[0, 160], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1)

    # --- Plot 2: Bland-Altman Plot ---
    fig.add_trace(go.Scatter(x=avg, y=diff, mode='markers', name='Difference', marker=dict(color='purple')), row=1, col=2)
    fig.add_hline(y=mean_diff, line_color="red", annotation_text=f"Mean Bias={mean_diff:.2f}", row=1, col=2)
    fig.add_hline(y=upper_loa, line_dash="dash", line_color="blue", annotation_text=f"Upper LoA={upper_loa:.2f}", row=1, col=2)
    fig.add_hline(y=lower_loa, line_dash="dash", line_color="blue", annotation_text=f"Lower LoA={lower_loa:.2f}", row=1, col=2)

    # --- Plot 3: Percent Bias Plot ---
    fig.add_trace(go.Scatter(x=x, y=percent_bias, mode='markers', name='% Bias', marker=dict(color='orange')), row=2, col=1)
    fig.add_hrect(y0=-15, y1=15, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=15, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-15, line_dash="dot", line_color="red", row=2, col=1)
    
    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Method Comparison Dashboard: R&D Lab vs QC Lab</b>',
        height=800,
        showlegend=False
    )
    fig.update_xaxes(title_text="R&D Lab (Reference)", row=1, col=1)
    fig.update_yaxes(title_text="QC Lab (Test)", row=1, col=1)
    fig.update_xaxes(title_text="Average of Methods", row=1, col=2)
    fig.update_yaxes(title_text="Difference (QC - R&D)", row=1, col=2)
    fig.update_xaxes(title_text="R&D Lab (Reference Concentration)", row=2, col=1)
    fig.update_yaxes(title_text="% Bias", range=[-25, 25], row=2, col=1)
    
    return fig, beta1_deming, beta0_deming, mean_diff, upper_loa, lower_loa

def plot_robustness_rsm():
    # --- Data Generation ---
    np.random.seed(42)
    # This simulates a Central Composite Design, which is excellent for RSM
    factors = {'Temp': [-1, 1, -1, 1, -1.414, 1.414, 0, 0, 0, 0, 0, 0, 0], 'pH': [-1, -1, 1, 1, 0, 0, -1.414, 1.414, 0, 0, 0, 0, 0]}
    df = pd.DataFrame(factors)
    # A more complex quadratic response surface model for a richer visualization
    df['Response'] = 95 - 5*df['Temp'] + 2*df['pH'] - 4*(df['Temp']**2) - 2*(df['pH']**2) + 3*df['Temp']*df['pH'] + np.random.normal(0, 1.5, len(df))
    
    # Fit the full quadratic model for RSM
    rsm_model = ols('Response ~ Temp + pH + I(Temp**2) + I(pH**2) + Temp:pH', data=df).fit()
    
    # --- Figure 1: World-Class Pareto Plot ---
    # We use a simpler screening model to get the main effects for the Pareto chart
    screening_model = ols('Response ~ Temp * pH', data=df).fit()
    effects = screening_model.params.iloc[1:].sort_values(key=abs, ascending=False)
    p_values = screening_model.pvalues.iloc[1:][effects.index]
    
    effect_data = pd.DataFrame({'Effect': effects.index, 'Value': effects.values, 'p_value': p_values})
    effect_data['color'] = np.where(effect_data['p_value'] < 0.05, 'salmon', 'skyblue') # Color by significance
    # A simplified significance threshold for visual effect
    significance_threshold = np.abs(effects.values).mean() * 1.5 

    fig_pareto = px.bar(
        effect_data, 
        x='Value', 
        y='Effect', 
        orientation='h', 
        title="<b>Pareto Plot of Factor Effects</b>",
        text=np.round(effect_data['Value'], 2),
        labels={'Value': 'Standardized Effect Magnitude', 'Effect': 'Factor or Interaction'},
        custom_data=['p_value']
    )
    fig_pareto.update_traces(
        marker_color=effect_data['color'],
        hovertemplate="<b>%{y}</b><br>Effect Value: %{x:.3f}<br>P-value: %{customdata[0]:.3f}<extra></extra>"
    )
    fig_pareto.add_vline(x=significance_threshold, line_dash="dash", line_color="red", annotation_text="Significance Threshold")
    fig_pareto.add_vline(x=-significance_threshold, line_dash="dash", line_color="red")
    fig_pareto.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)

    # --- Figures 2 & 3: World-Class RSM Plots ---
    temp_range = np.linspace(-2, 2, 50); ph_range = np.linspace(-2, 2, 50)
    grid_temp, grid_ph = np.meshgrid(temp_range, ph_range)
    grid_df = pd.DataFrame({'Temp': grid_temp.ravel(), 'pH': grid_ph.ravel()})
    grid_df['Predicted_Response'] = rsm_model.predict(grid_df)
    predicted_response_grid = grid_df['Predicted_Response'].values.reshape(50, 50)
    
    # Find optimal point
    opt_idx = grid_df['Predicted_Response'].idxmax()
    opt_temp, opt_ph, opt_response = grid_df.loc[opt_idx]

    # --- Figure 2: Publication-Quality 2D Contour Plot ---
    fig_contour = go.Figure(data=go.Contour(
        z=predicted_response_grid, x=temp_range, y=ph_range, 
        colorscale='Viridis',
        contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=12, color='white')),
        line=dict(width=2),
        hoverinfo='x+y+z'
    ))
    fig_contour.add_trace(go.Scatter(
        x=df['Temp'], y=df['pH'], mode='markers', 
        marker=dict(color='white', size=10, symbol='x', line=dict(color='black', width=2)), 
        name='Design Points',
        hovertemplate="Temp: %{x:.2f}<br>pH: %{y:.2f}<extra></extra>"
    ))
    fig_contour.add_trace(go.Scatter(
        x=[opt_temp], y=[opt_ph], mode='markers',
        marker=dict(color='red', size=16, symbol='star', line=dict(color='white', width=2)),
        name='Optimal Point',
        hovertemplate=f"<b>Optimal Point</b><br>Temp: {opt_temp:.2f}<br>pH: {opt_ph:.2f}<br>Response: {opt_response:.2f}<extra></extra>"
    ))
    fig_contour.update_layout(
        title="<b>2D Contour Plot of Response Surface</b>", 
        xaxis_title="Temperature (coded units)", 
        yaxis_title="pH (coded units)",
        title_x=0.5,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # --- Figure 3: Immersive 3D Surface Plot ---
    fig_surface = go.Figure(data=[go.Surface(
        z=predicted_response_grid, x=temp_range, y=ph_range, 
        colorscale='Viridis',
        contours = {
            "x": {"show": True, "start": -2, "end": 2, "size": 0.5, "color":"white"},
            "y": {"show": True, "start": -2, "end": 2, "size": 0.5, "color":"white"},
            "z": {"show": True, "start": predicted_response_grid.min(), "end": predicted_response_grid.max(), "size": 5}
        },
        hoverinfo='x+y+z'
    )])
    fig_surface.add_trace(go.Scatter3d(
        x=df['Temp'], y=df['pH'], z=df['Response'], 
        mode='markers', 
        marker=dict(color='red', size=5, symbol='diamond'), 
        name='Design Points'
    ))
    fig_surface.add_trace(go.Scatter3d(
        x=[opt_temp], y=[opt_ph], z=[opt_response],
        mode='markers',
        # CORRECTED LINE: Changed symbol from 'star' to 'diamond'
        marker=dict(color='yellow', size=10, symbol='diamond'),
        name='Optimal Point'
    ))
    fig_surface.update_layout(
        title='<b>3D Response Surface Plot</b>', 
        scene=dict(
            xaxis_title="Temperature", 
            yaxis_title="pH", 
            zaxis_title="Assay Response",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title_x=0.5
    )
    
    return fig_pareto, fig_contour, fig_surface, effects
    
def plot_shewhart():
    # --- Data Generation ---
    np.random.seed(42)
    in_control_data = np.random.normal(loc=100.0, scale=2.0, size=15)
    reagent_shift_data = np.random.normal(loc=108.0, scale=2.0, size=10)
    data = np.concatenate([in_control_data, reagent_shift_data])
    x = np.arange(1, len(data) + 1)

    # --- Calculations ---
    mean = np.mean(data[:15])
    mr = np.abs(np.diff(data))
    mr_mean = np.mean(mr[:14])
    sigma_est = mr_mean / 1.128
    UCL_I, LCL_I = mean + 3 * sigma_est, mean - 3 * sigma_est
    UCL_MR = mr_mean * 3.267
    
    # Identify out-of-control points
    out_of_control_I_idx = np.where((data > UCL_I) | (data < LCL_I))[0]

    # --- Figure Creation ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        subplot_titles=("<b>I-Chart: Monitors Accuracy (Bias)</b>", "<b>MR-Chart: Monitors Precision (Variability)</b>"), 
        vertical_spacing=0.1, row_heights=[0.7, 0.3]
    )

    # --- I-Chart Construction ---
    # Add Shaded Zones for context
    for i, color in zip([1, 2, 3], ['#a5d6a7', '#fff59d', '#ef9a9a']): # Green, Yellow, Red
        fig.add_hrect(y0=mean - i*sigma_est, y1=mean + i*sigma_est, fillcolor=color, opacity=0.3, layer="below", line_width=0, row=1, col=1)
    
    # Add Center and Control Lines with annotations
    fig.add_hline(y=mean, line=dict(dash='dash', color='black'), annotation_text=f"Mean={mean:.1f}", row=1, col=1)
    fig.add_hline(y=UCL_I, line=dict(color='red'), annotation_text=f"UCL={UCL_I:.1f}", row=1, col=1)
    fig.add_hline(y=LCL_I, line=dict(color='red'), annotation_text=f"LCL={LCL_I:.1f}", row=1, col=1)

    # Plot the SINGLE, CONTINUOUS line for the process data
    fig.add_trace(go.Scatter(
        x=x, y=data, 
        mode='lines+markers', name='Control Value',
        line=dict(color='royalblue'),
        marker=dict(color='royalblue', size=6),
        hovertemplate="Run %{x}<br>Value: %{y:.2f}<extra></extra>"
    ), row=1, col=1)

    # Add ONLY the out-of-control markers on top
    fig.add_trace(go.Scatter(
        x=x[out_of_control_I_idx], y=data[out_of_control_I_idx], 
        mode='markers', name='Out of Control Signal',
        marker=dict(color='red', size=12, symbol='x-thin', line=dict(width=3)),
        hovertemplate="<b>VIOLATION</b><br>Run %{x}<br>Value: %{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    # Add annotations for violations
    for idx in out_of_control_I_idx:
        fig.add_annotation(x=x[idx], y=data[idx], text="Rule 1 Violation", showarrow=True, arrowhead=2, ax=20, ay=-40, row=1, col=1, font=dict(color="red"))
        
    # Highlight the process shift event
    fig.add_vrect(x0=15.5, x1=25.5, fillcolor="rgba(255,165,0,0.2)", layer="below", line_width=0, 
                  annotation_text="New Reagent Lot", annotation_position="top left", row=1, col=1)

    # --- MR-Chart Construction ---
    fig.add_trace(go.Scatter(
        x=x[1:], y=mr, mode='lines+markers', name='Moving Range', 
        line=dict(color='teal'),
        hovertemplate="Range (Run %{x}-%{x_prev})<br>Value: %{y:.2f}<extra></extra>".replace('%{x_prev}', str(list(x[:-1])))
    ), row=2, col=1)
    fig.add_hline(y=mr_mean, line=dict(dash='dash', color='black'), annotation_text=f"Mean={mr_mean:.1f}", row=2, col=1)
    fig.add_hline(y=UCL_MR, line=dict(color='red'), annotation_text=f"UCL={UCL_MR:.1f}", row=2, col=1)

    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Process Stability Monitoring: Shewhart I-MR Chart</b>', 
        height=800, 
        showlegend=False,
        margin=dict(t=100)
    )
    fig.update_yaxes(title_text="Concentration (ng/mL)", row=1, col=1)
    fig.update_yaxes(title_text="Range (ng/mL)", row=2, col=1)
    fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    
    return fig
def plot_ewma_cusum(chart_type, lmbda, k_sigma, H_sigma):
    # --- Data Generation ---
    np.random.seed(101)
    in_control_data = np.random.normal(50, 2, 25)
    shift_data = np.random.normal(52.5, 2, 15) # A small 1.25-sigma shift
    data = np.concatenate([in_control_data, shift_data])
    target = np.mean(in_control_data)
    sigma = np.std(in_control_data, ddof=1)
    x_axis = np.arange(1, len(data) + 1)

    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("<b>Raw Process Data</b>", f"<b>{chart_type} Chart</b>"),
        vertical_spacing=0.1
    )

    # --- Plot 1: Raw Data ---
    fig.add_trace(go.Scatter(
        x=x_axis, y=data, mode='lines+markers', name='Daily Control',
        marker=dict(color='grey'), line=dict(color='lightgrey'),
        hovertemplate="Run %{x}<br>Value: %{y:.2f}<extra></extra>"
    ), row=1, col=1)
    fig.add_hline(y=target, line_dash="dash", line_color="black", annotation_text=f"Target Mean={target:.1f}", row=1, col=1)
    fig.add_vrect(x0=25.5, x1=40.5, fillcolor="orange", opacity=0.2, layer="below", line_width=0,
                  annotation_text="1.25σ Shift Introduced", annotation_position="top left", row=1, col=1)
    
    # --- Plot 2: EWMA or CUSUM Chart ---
    if chart_type == 'EWMA':
        ewma_vals = np.zeros_like(data); ewma_vals[0] = target
        for i in range(1, len(data)):
            ewma_vals[i] = lmbda * data[i] + (1 - lmbda) * ewma_vals[i-1]
        
        L = 3
        UCL = [target + L * sigma * np.sqrt((lmbda / (2 - lmbda)) * (1 - (1 - lmbda)**(2 * i))) for i in range(1, len(data) + 1)]
        out_idx = np.where(ewma_vals > UCL)[0]
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=ewma_vals, mode='lines+markers', name=f'EWMA (λ={lmbda})',
            line=dict(color='purple'), hovertemplate="Run %{x}<br>EWMA: %{y:.2f}<extra></extra>"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=UCL, mode='lines', name='EWMA UCL', line=dict(color='red', dash='dash')
        ), row=2, col=1)
        
        if len(out_idx) > 0:
            signal_idx = out_idx[0]
            fig.add_trace(go.Scatter(
                x=[x_axis[signal_idx]], y=[ewma_vals[signal_idx]], mode='markers',
                marker=dict(color='red', size=15, symbol='x'), name='Signal'
            ), row=2, col=1)
            fig.add_annotation(x=x_axis[signal_idx], y=ewma_vals[signal_idx], text="<b>Signal!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, row=2, col=1)
        
        fig.update_yaxes(title_text="EWMA Value", row=2, col=1)

    else: # CUSUM
        k = k_sigma * sigma; H = H_sigma * sigma
        SH, SL = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)):
            SH[i] = max(0, SH[i-1] + (data[i] - target) - k)
            SL[i] = max(0, SL[i-1] + (target - data[i]) - k)
        out_idx = np.where((SH > H) | (SL > H))[0]
        
        fig.add_trace(go.Scatter(x=x_axis, y=SH, mode='lines+markers', name='High-Side CUSUM (SH)', line=dict(color='darkcyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=SL, mode='lines+markers', name='Low-Side CUSUM (SL)', line=dict(color='darkorange')), row=2, col=1)
        fig.add_hline(y=H, line_dash="dash", line_color="red", annotation_text=f"Limit H={H:.1f}", row=2, col=1)
        
        if len(out_idx) > 0:
            signal_idx = out_idx[0]
            fig.add_trace(go.Scatter(x=[x_axis[signal_idx]], y=[SH[signal_idx]], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Signal'), row=2, col=1)
            fig.add_annotation(x=x_axis[signal_idx], y=SH[signal_idx], text="<b>Signal!</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, row=2, col=1)

        fig.update_yaxes(title_text="Cumulative Sum", row=2, col=1)
        
    # --- Final Layout Updates ---
    fig.update_layout(
        title_text=f'<b>Small Shift Detection Dashboard ({chart_type})</b>',
        height=800,
        showlegend=False
    )
    fig.update_yaxes(title_text="Assay Response", row=1, col=1)
    fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    
    return fig

def plot_multi_rule():
    # --- Data Generation to trigger specific Westgard rules ---
    np.random.seed(3)
    mean, std = 100, 2
    data = np.array([
        100.5, 99.8, 101.2, 98.9, 100.2, # In control
        104.5, 105.1, # 2_2s violation
        100.1, 99.5, 
        102.3, 102.8, 103.1, 102.5, # 4_1s violation
        99.9,
        106.5, # 1_3s violation
        100.8, 98.5,
        104.2, 95.5, # R_4s violation
        100.0
    ])
    x = np.arange(1, len(data) + 1)
    z_scores = (data - mean) / std

    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("<b>Levey-Jennings Chart with Westgard Violations</b>", "<b>Distribution of QC Data</b>"),
        vertical_spacing=0.15, row_heights=[0.7, 0.3]
    )

    # --- Plot 1: Levey-Jennings Chart ---
    # Add Shaded Zones
    for i, color in zip([3, 2, 1], ['#ef9a9a', '#fff59d', '#a5d6a7']): # Red, Yellow, Green
        fig.add_hrect(y0=mean - i*std, y1=mean + i*std, fillcolor=color, opacity=0.3, layer="below", line_width=0, row=1, col=1)
    
    # Add Center and Control Lines
    for i in [-3, -2, -1, 1, 2, 3]:
        fig.add_hline(y=mean + i*std, line_dash="dot", line_color="gray", annotation_text=f"{i}s", row=1, col=1)
    fig.add_hline(y=mean, line_dash="dash", line_color="black", annotation_text="Mean", row=1, col=1)

    # Plot the continuous QC data line
    fig.add_trace(go.Scatter(
        x=x, y=data, mode='lines+markers', name='QC Sample',
        line=dict(color='darkblue'),
        hovertemplate="Run: %{x}<br>Value: %{y:.2f}<br>Z-Score: %{customdata:.2f}s<extra></extra>",
        customdata=z_scores
    ), row=1, col=1)

    # --- Identify and Annotate Violations ---
    violations = []
    if np.any(np.abs(z_scores) > 3): # 1_3s
        idx = np.where(np.abs(z_scores) > 3)[0][0]
        violations.append({'x': x[idx], 'y': data[idx], 'rule': '1_3s Violation'})
    for i in range(1, len(z_scores)): # 2_2s
        if (z_scores[i] > 2 and z_scores[i-1] > 2) or (z_scores[i] < -2 and z_scores[i-1] < -2):
            violations.append({'x': x[i], 'y': data[i], 'rule': '2_2s Violation'})
    for i in range(3, len(z_scores)): # 4_1s
        if np.all(z_scores[i-3:i+1] > 1) or np.all(z_scores[i-3:i+1] < -1):
             violations.append({'x': x[i], 'y': data[i], 'rule': '4_1s Violation'})
    for i in range(1, len(z_scores)): # R_4s
        if (z_scores[i] > 2 and z_scores[i-1] < -2) or (z_scores[i] < -2 and z_scores[i-1] > 2):
            violations.append({'x': x[i], 'y': data[i], 'rule': 'R_4s Violation'})
    
    violation_points = pd.DataFrame(violations)
    if not violation_points.empty:
        fig.add_trace(go.Scatter(
            x=violation_points['x'], y=violation_points['y'], mode='markers', name='Violation',
            marker=dict(color='red', size=15, symbol='x-thin', line=dict(width=3))
        ), row=1, col=1)
        for _, row in violation_points.iterrows():
            fig.add_annotation(x=row['x'], y=row['y'], text=f"<b>{row['rule']}</b>", showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="red"), row=1, col=1)

    # --- Plot 2: Distribution of QC Data ---
    fig.add_trace(go.Histogram(x=data, name='Distribution', histnorm='probability density', marker_color='darkblue'), row=2, col=1)
    x_norm = np.linspace(mean - 4*std, mean + 4*std, 100)
    y_norm = stats.norm.pdf(x_norm, mean, std)
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Curve', line=dict(color='red', dash='dash')), row=2, col=1)

    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>QC Run Validation Dashboard</b>',
        height=800,
        showlegend=False
    )
    fig.update_yaxes(title_text="Measured Value", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1)
    
    return fig


def plot_capability(scenario):
    np.random.seed(42); LSL, USL = 90, 110
    if scenario == 'Ideal': data = np.random.normal(100, (USL-LSL)/(6*1.67), 200)
    elif scenario == 'Shifted': data = np.random.normal(105, (USL-LSL)/(6*1.67), 200)
    elif scenario == 'Variable': data = np.random.normal(100, (USL-LSL)/(6*0.9), 200)
    else: data = np.concatenate([np.random.normal(97, 2, 100), np.random.normal(103, 2, 100)])
    sigma_hat = np.std(data, ddof=1); Cpu = (USL - data.mean()) / (3 * sigma_hat); Cpl = (data.mean() - LSL) / (3 * sigma_hat); Cpk = np.min([Cpu, Cpl])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Process Control (I-Chart)", "Process Capability (Histogram)"), vertical_spacing=0.1, row_heights=[0.4, 0.6])
    x_axis = np.arange(1, len(data) + 1); mean_i = data.mean(); mr = np.abs(np.diff(data)); mr_mean = np.mean(mr); UCL_I, LCL_I = mean_i + 3*(mr_mean/1.128), mean_i - 3*(mr_mean/1.128);
    fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines', line=dict(color='lightgrey'), name='Control Value'), row=1, col=1)
    out_of_control_idx = np.where((data > UCL_I) | (data < LCL_I))[0]
    fig.add_trace(go.Scatter(x=x_axis[out_of_control_idx], y=data[out_of_control_idx], mode='markers', marker=dict(color='red', size=8), name='Signal'), row=1, col=1)
    fig.add_hline(y=mean_i, line_dash="dash", line_color="black", row=1, col=1); fig.add_hline(y=UCL_I, line_color="red", row=1, col=1); fig.add_hline(y=LCL_I, line_color="red", row=1, col=1);
    fig_hist = px.histogram(data, nbins=30, histnorm='probability density'); fig.add_trace(fig_hist.data[0], row=2, col=1)
    fig.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL", row=2, col=1); fig.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL", row=2, col=1); fig.add_vline(x=data.mean(), line_dash="dot", line_color="black", annotation_text="Mean", row=2, col=1)
    color = "darkgreen" if Cpk >= 1.33 and scenario != 'Out of Control' else "darkred"; text = f"Cpk = {Cpk:.2f}" if scenario != 'Out of Control' else "Cpk: INVALID"
    fig.add_annotation(text=text, align='left', showarrow=False, xref='paper', yref='paper', x=0.05, y=0.45, bordercolor="black", borderwidth=1, bgcolor=color, font=dict(color="white"))
    fig.update_layout(title_text=f'Process Capability Analysis - Scenario: {scenario}', height=800, showlegend=False); return fig, Cpk, scenario

def plot_anomaly_detection():
    # --- Data Generation ---
    np.random.seed(42)
    X_normal = np.random.multivariate_normal([100, 20], [[5, 2],[2, 1]], 200)
    X_anomalies = np.array([[95, 25], [110, 18], [115, 28]])
    X = np.vstack([X_normal, X_anomalies])
    
    # --- Model Training ---
    model = IsolationForest(n_estimators=100, contamination=0.015, random_state=42)
    model.fit(X)
    y_pred = model.predict(X) # -1 for anomalies, 1 for normal
    
    # --- Figure Creation ---
    # Create the background contour plot
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-5, X[:, 0].max()+5, 100), np.linspace(X[:, 1].min()-5, X[:, 1].max()+5, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig = go.Figure()

    # Add the contour trace for the decision boundary
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:,0], z=Z, 
        colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(0, 0, 255, 0.2)']], # Red for anomaly, blue for normal
        showscale=False,
        hoverinfo='none'
    ))
    
    # Separate normal and anomaly points for plotting
    df_plot = pd.DataFrame(X, columns=['x', 'y'])
    df_plot['status'] = ['Anomaly' if p == -1 else 'Normal' for p in y_pred]
    
    # Add Normal Points Trace
    normal_df = df_plot[df_plot['status'] == 'Normal']
    fig.add_trace(go.Scatter(
        x=normal_df['x'], y=normal_df['y'],
        mode='markers',
        marker=dict(color='royalblue', size=8, line=dict(width=1, color='black')),
        name='Normal Run',
        hovertemplate="<b>Status: Normal</b><br>Response: %{x:.2f}<br>Time: %{y:.2f}<extra></extra>"
    ))
    
    # Add Anomaly Points Trace
    anomaly_df = df_plot[df_plot['status'] == 'Anomaly']
    fig.add_trace(go.Scatter(
        x=anomaly_df['x'], y=anomaly_df['y'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x-thin', line=dict(width=3)),
        name='Anomaly',
        hovertemplate="<b>Status: Anomaly</b><br>Response: %{x:.2f}<br>Time: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title_text='<b>Multivariate Anomaly Detection (Isolation Forest)</b>',
        xaxis_title='Assay Response (Fluorescence Units)',
        yaxis_title='Incubation Time (min)',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title_x=0.5
    )
    return fig


def plot_predictive_qc():
    # --- Data Generation ---
    np.random.seed(1)
    n_points = 150
    X1 = np.random.normal(5, 2, n_points) # Reagent Age (days)
    X2 = np.random.normal(25, 3, n_points) # Incubation Temp (C)
    # Create a linear combination that predicts failure
    logit_p = -15 + 1.5 * X1 + 0.5 * X2 + np.random.normal(0, 2, n_points)
    p = 1 / (1 + np.exp(-logit_p))
    y = np.random.binomial(1, p)
    X = np.vstack([X1, X2]).T
    
    # --- Model Training and Prediction ---
    model = LogisticRegression().fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Decision Boundary Risk Map</b>", "<b>Model Performance: Probability Distributions</b>"),
        column_widths=[0.6, 0.4]
    )

    # --- Plot 1: Decision Boundary Risk Map ---
    # Create grid for contour plot
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
    )
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    
    # Add contour trace
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:,0], z=Z,
        colorscale='RdYlGn_r',
        colorbar=dict(title="P(Fail)"),
        showscale=True,
        hoverinfo='none'
    ), row=1, col=1)
    
    # Add scatter plot of actual data
    df_plot = pd.DataFrame(X, columns=['Reagent Age', 'Incubation Temp'])
    df_plot['Outcome'] = ['Pass' if i == 0 else 'Fail' for i in y]
    df_plot['P(Fail)'] = probabilities
    
    fig_scatter = px.scatter(
        df_plot, x='Reagent Age', y='Incubation Temp', color='Outcome',
        color_discrete_map={'Pass':'green', 'Fail':'red'},
        symbol='Outcome', symbol_map={'Pass': 'circle', 'Fail': 'x'},
        custom_data=['P(Fail)']
    )
    for trace in fig_scatter.data:
        trace.update(hovertemplate="<b>%{customdata[0]:.1%} P(Fail)</b><br>Age: %{x:.1f} days<br>Temp: %{y:.1f}°C<extra></extra>")
        fig.add_trace(trace, row=1, col=1)

    # --- Plot 2: Probability Distributions ---
    fig.add_trace(go.Histogram(
        x=df_plot[df_plot['Outcome'] == 'Pass']['P(Fail)'],
        name='Actual Pass', histnorm='probability density',
        marker_color='green', opacity=0.7
    ), row=1, col=2)
    fig.add_trace(go.Histogram(
        x=df_plot[df_plot['Outcome'] == 'Fail']['P(Fail)'],
        name='Actual Fail', histnorm='probability density',
        marker_color='red', opacity=0.7
    ), row=1, col=2)
    
    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>Predictive QC Dashboard: Identifying At-Risk Runs</b>',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        barmode='overlay',
        title_x=0.5
    )
    fig.update_xaxes(title_text="Reagent Age (days)", row=1, col=1)
    fig.update_yaxes(title_text="Incubation Temp (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Probability of Failure", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    
    return fig

def plot_forecasting():
    # --- Data Generation ---
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W'))
    # Trend with a changepoint
    trend1 = np.linspace(0, 2, 52)
    trend2 = np.linspace(2.1, 8, 52)
    trend = np.concatenate([trend1, trend2])
    seasonality = 1.5 * np.sin(np.arange(104) * (2 * np.pi / 52.14)) # Annual seasonality
    noise = np.random.normal(0, 0.5, 104)
    y = 50 + trend + seasonality + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    # --- Model Training and Prediction ---
    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.5)
    model.fit(df)
    future = model.make_future_dataframe(periods=26, freq='W')
    forecast = model.predict(future)
    
    # --- Figure 1: Main Forecast Plot ---
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(
        title_text='<b>Control Performance Forecast vs. Specification Limit</b>',
        xaxis_title='Date', yaxis_title='Control Value (U/mL)',
        showlegend=True
    )
    # Add spec limit and highlight breaches
    spec_limit = 58
    fig1.add_hline(y=spec_limit, line_dash="dash", line_color="red", annotation_text="Upper Spec Limit")
    breaches = forecast[forecast['yhat_upper'] > spec_limit]
    if not breaches.empty:
        fig1.add_trace(go.Scatter(
            x=breaches['ds'], y=breaches['yhat'],
            mode='markers', name='Predicted Breach',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
    # Add forecast horizon annotation
    fig1.add_vrect(x0=forecast['ds'].iloc[-26], x1=forecast['ds'].iloc[-1], 
                  fillcolor="rgba(0,100,80,0.1)", layer="below", line_width=0,
                  annotation_text="Forecast Horizon", annotation_position="top left")

    # --- Figure 2: Trend & Changepoints Plot (CORRECTED IMPLEMENTATION) ---
    fig2 = go.Figure()
    # Plot the trend line
    fig2.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['trend'], 
        mode='lines', name='Trend', line=dict(color='navy')
    ))
    
    # Manually add changepoints to the Plotly figure
    if len(model.changepoints) > 0:
        signif_changepoints = model.changepoints[
            np.abs(np.nanmean(model.params['delta'], axis=0)) >= 0.01
        ]
        if len(signif_changepoints) > 0:
            for cp in signif_changepoints:
                # CORRECTED LINE: Changed 'dashed' to 'dash'
                fig2.add_vline(x=cp, line_width=1, line_dash="dash", line_color="red")

    fig2.update_layout(
        title_text='<b>Decomposed Trend with Detected Changepoints</b>',
        xaxis_title='Date', yaxis_title='Trend Value'
    )
    
    # --- Figure 3: Seasonality Plot ---
    # Prophet's plot_components_plotly generates a figure with multiple subplots.
    # We will extract only the yearly seasonality for a cleaner look.
    fig3_full = plot_components_plotly(model, forecast, figsize=(900, 200))
    
    # Create a new, clean figure just for the desired component
    fig3 = go.Figure()
    for trace in fig3_full.select_traces(selector=dict(xaxis='x2')): # x2 is typically the yearly component
        fig3.add_trace(trace)
    
    fig3.update_layout(
        title_text='<b>Decomposed Yearly Seasonal Effect</b>',
        xaxis_title='Day of Year',
        yaxis_title='Seasonal Component',
        showlegend=False
    )

    return fig1, fig2, fig3

def plot_wilson(successes, n_samples):
    # --- Data for CI Comparison Plot ---
    p_hat = successes / n_samples if n_samples > 0 else 0
    wald_lower, wald_upper = stats.norm.interval(0.95, loc=p_hat, scale=np.sqrt(p_hat*(1-p_hat)/n_samples)) if n_samples > 0 else (0,0)
    wilson_lower, wilson_upper = wilson_score_interval(p_hat, n_samples)
    cp_lower, cp_upper = stats.beta.interval(0.95, successes, n_samples - successes + 1) if n_samples > 0 else (0,1)
    
    intervals = {
        "Wald (Approximate)": (wald_lower, wald_upper, 'red'),
        "Wilson Score": (wilson_lower, wilson_upper, 'blue'),
        "Clopper-Pearson (Exact)": (cp_lower, cp_upper, 'green')
    }
    
    # --- Figure 1: CI Comparison ---
    fig1 = go.Figure()
    for name, (lower, upper, color) in intervals.items():
        fig1.add_trace(go.Scatter(
            x=[p_hat], y=[name],
            error_x=dict(type='data', array=[upper-p_hat], arrayminus=[p_hat-lower]),
            mode='markers',
            marker=dict(color=color, size=12),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Observed: {p_hat:.2%}<br>Lower: {lower:.3f}<br>Upper: {upper:.3f}<extra></extra>"
        ))
    
    fig1.add_vrect(x0=0.9, x1=1.0, fillcolor="rgba(0,255,0,0.1)", layer="below", line_width=0, annotation_text="Target Zone > 90%", annotation_position="bottom left")
    fig1.update_layout(
        title_text=f'<b>Comparing 95% CIs for {successes}/{n_samples} Concordant Results</b>',
        xaxis_title='Concordance Rate',
        showlegend=False,
        height=500,
        xaxis_range=[-0.05, 1.05]
    )

    # --- Figure 2: Coverage Probability ---
    true_proportions = np.linspace(0.01, 0.99, 200)
    n_coverage = n_samples
    
    # Calculate coverage probabilities
    @st.cache_data
    def calculate_coverage(n_cov, p_array):
        wald_cov = []
        wilson_cov = []
        cp_cov = []
        for p in p_array:
            k = np.arange(0, n_cov + 1)
            p_k = stats.binom.pmf(k, n_cov, p)
            
            # Wald coverage
            wald_l, wald_u = stats.norm.interval(0.95, loc=k/n_cov, scale=np.sqrt((k/n_cov)*(1-k/n_cov)/n_cov))
            wald_cov.append(np.sum(p_k[(wald_l <= p) & (p <= wald_u)]))
            
            # Wilson coverage
            wilson_l, wilson_u = wilson_score_interval(k/n_cov, n_cov)
            wilson_cov.append(np.sum(p_k[(wilson_l <= p) & (p <= wilson_u)]))
            
            # Clopper-Pearson coverage
            cp_l, cp_u = stats.beta.interval(0.95, k, n_cov - k + 1)
            cp_cov.append(np.sum(p_k[(cp_l <= p) & (p <= cp_u)]))
            
        return wald_cov, wilson_cov, cp_cov
        
    wald_coverage, wilson_coverage, cp_coverage = calculate_coverage(n_coverage, true_proportions)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=true_proportions, y=wald_coverage, mode='lines', name='Wald', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=true_proportions, y=wilson_coverage, mode='lines', name='Wilson Score', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=true_proportions, y=cp_coverage, mode='lines', name='Clopper-Pearson', line=dict(color='green')))
    
    fig2.add_hrect(y0=0, y1=0.95, fillcolor="rgba(255,0,0,0.1)", layer="below", line_width=0)
    fig2.add_hline(y=0.95, line_dash="dash", line_color="black", annotation_text="Nominal 95% Coverage")
    
    fig2.update_layout(
        title_text=f'<b>Coverage Probability for n={n_samples}</b>',
        xaxis_title='True Proportion',
        yaxis_title='Actual Coverage Probability',
        yaxis_range=[min(0.8, np.nanmin(wald_coverage)-0.02 if np.any(np.isfinite(wald_coverage)) else 0.8), 1.02],
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )

    return fig1, fig2

def plot_bayesian(prior_type):
    # --- Data and Prior Definition ---
    n_qc, successes_qc = 20, 18
    observed_rate = successes_qc / n_qc
    
    if prior_type == "Strong R&D Prior":
        prior_alpha, prior_beta = 490, 10 # Corresponds to 490 successes in 500 trials
    elif prior_type == "Skeptical/Regulatory Prior":
        prior_alpha, prior_beta = 10, 10 # Prefers a fair coin, requires strong evidence to move
    else: # "No Prior (Frequentist)"
        prior_alpha, prior_beta = 1, 1 # A flat, uninformative prior
    
    p_range = np.linspace(0.6, 1.0, 501)
    
    # --- Calculate the Three Distributions ---
    # 1. Prior Distribution
    prior_dist = beta.pdf(p_range, prior_alpha, prior_beta)
    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    
    # 2. Likelihood Distribution (from new data)
    likelihood = stats.binom.pmf(k=successes_qc, n=n_qc, p=p_range)
    
    # 3. Posterior Distribution
    posterior_alpha, posterior_beta = prior_alpha + successes_qc, prior_beta + (n_qc - successes_qc)
    posterior_dist = beta.pdf(p_range, posterior_alpha, posterior_beta)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # --- Create the World-Class Plot ---
    fig = go.Figure()
    
    # Normalize for clean visualization
    max_y = np.max(posterior_dist) * 1.1
    
    # Plot Likelihood first (as the evidence)
    fig.add_trace(go.Scatter(
        x=p_range, y=likelihood * max_y / np.max(likelihood),
        mode='lines', name='Likelihood (from QC Data)',
        line=dict(dash='dot', color='red', width=2),
        fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)',
        hovertemplate="p=%{x:.3f}<br>Likelihood (scaled)<extra></extra>"
    ))
    
    # Plot Prior second (as the old belief)
    fig.add_trace(go.Scatter(
        x=p_range, y=prior_dist,
        mode='lines', name='Prior Belief',
        line=dict(dash='dash', color='green', width=3),
        hovertemplate="p=%{x:.3f}<br>Prior Density: %{y:.2f}<extra></extra>"
    ))
    
    # Plot Posterior last (as the final result)
    fig.add_trace(go.Scatter(
        x=p_range, y=posterior_dist,
        mode='lines', name='Posterior Belief',
        line=dict(color='blue', width=4),
        fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)',
        hovertemplate="p=%{x:.3f}<br>Posterior Density: %{y:.2f}<extra></extra>"
    ))

    # Add annotations for the means/peaks to tell the story
    fig.add_vline(x=prior_mean, line_dash="dash", line_color="green", annotation_text=f"Prior Mean={prior_mean:.3f}")
    fig.add_vline(x=observed_rate, line_dash="dot", line_color="red", annotation_text=f"Data (MLE)={observed_rate:.3f}")
    fig.add_vline(x=posterior_mean, line_dash="solid", line_color="blue", annotation_text=f"Posterior Mean={posterior_mean:.3f}", annotation_font=dict(color="blue", size=14))
    
    fig.update_layout(
        title_text='<b>Bayesian Inference: How Evidence Updates Belief</b>',
        xaxis_title='Assay Pass Rate (Concordance)',
        yaxis_title='Probability Density / Scaled Likelihood',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title_x=0.5
    )
    
    return fig, prior_mean, observed_rate, posterior_mean
    
    return fig, prior_mean, observed_rate, posterior_mean

def plot_ci_concept(n=30):
    # --- Data Generation ---
    np.random.seed(123)
    pop_mean, pop_std = 100, 15
    n_sims = 100
    
    # Generate population for visualization
    population = np.random.normal(pop_mean, pop_std, 10000)
    
    # Generate many sample means for the sampling distribution based on the selected n
    sample_means = [np.mean(np.random.normal(pop_mean, pop_std, n)) for _ in range(1000)]
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("<b>The Theoretical Universe: Population vs. Sampling Distribution</b>", f"<b>The Practical Result: 100 Simulated CIs with n={n}</b>"),
        vertical_spacing=0.15
    )
    
    # --- Plot 1: Distributions ---
    # Population Distribution
    fig.add_trace(go.Histogram(
        x=population, histnorm='probability density', name='True Population',
        marker_color='skyblue', opacity=0.6,
        hovertemplate="Value: %{x}<br>Density: %{y}<extra></extra>"
    ), row=1, col=1)
    # Sampling Distribution of the Mean
    fig.add_trace(go.Histogram(
        x=sample_means, histnorm='probability density', name=f'Distribution of Sample Means (n={n})',
        marker_color='darkorange', opacity=0.6,
        hovertemplate="Sample Mean: %{x:.2f}<br>Density: %{y}<extra></extra>"
    ), row=1, col=1)
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}", row=1, col=1)

    # --- Plot 2: Confidence Interval Simulation ---
    capture_count = 0
    total_width = 0
    for i in range(n_sims):
        sample = np.random.normal(pop_mean, pop_std, n)
        sample_mean = np.mean(sample)
        # Using known pop_std for simplicity in this conceptual plot
        margin_of_error = 1.96 * (pop_std / np.sqrt(n))
        ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error
        total_width += (ci_upper - ci_lower)
        
        color = 'cornflowerblue' if ci_lower <= pop_mean <= ci_upper else 'red'
        if color == 'cornflowerblue':
            capture_count += 1
        
        status = "Capture" if color == 'cornflowerblue' else "Miss"
        
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper], y=[i, i], mode='lines',
            line=dict(color=color, width=3),
            hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Status: {status}<br>Interval: [{ci_lower:.2f}, {ci_upper:.2f}]<extra></extra>"
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[sample_mean], y=[i], mode='markers',
            marker=dict(color='black', size=5, symbol='line-ns-open'),
            hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Sample Mean: {sample_mean:.2f}<extra></extra>"
        ), row=2, col=1)
    
    avg_width = total_width / n_sims
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}", row=2, col=1)
    
    # --- Final Layout Updates ---
    fig.update_layout(
        title_text='<b>The Confidence Interval Concept: From Theory to Practice</b>',
        height=900,
        showlegend=False,
        barmode='overlay'
    )
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Simulation Run", range=[-2, n_sims+2], row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    
    # Return two separate figures for individual plotting in the main app
    fig1 = go.Figure(data=fig.data[0:3])
    fig1.update_layout(title_text=f"<b>Theoretical Universe (Sample Size n={n})</b>", yaxis_title="Density", xaxis_title="Value", showlegend=True, barmode='overlay')
    
    fig2 = go.Figure(data=fig.data[3:])
    fig2.update_layout(title_text=f"<b>Practical Result: 100 Simulated CIs (Sample Size n={n})</b>", yaxis_title="Simulation Run", xaxis_title="Value", showlegend=False, yaxis_range=[-2, n_sims+2])
    
    return fig1, fig2, capture_count, n_sims, avg_width

# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
st.title("🔬 An Interactive Guide to Assay Transfer Statistics")
st.markdown("Welcome to this interactive guide. It's a collection of tools to help explore the statistical methods that support a robust assay transfer and lifecycle management plan, bridging classical SPC with modern ML/AI concepts.")

st.plotly_chart(create_conceptual_map_plotly(), use_container_width=True)
st.markdown("This map illustrates how foundational **Academic Disciplines** like Statistics and Industrial Engineering give rise to **Core Domains** such as Statistical Process Control (SPC) and Statistical Inference. These domains, in turn, provide the **Sub-Domains & Concepts** that are the basis for the **Specific Tools & Applications** you can explore in this guide. Use the sidebar to navigate through these practical applications.")
st.divider()

st.header("The Scientist's Journey: A Three-Act Story")
st.markdown("""In the world of quality and development, our story always has the same **Hero**: the dedicated scientist, engineer, or analyst. And it always has the same **Villain**: insidious, hidden, and costly **Variation**.
This toolkit is structured as a three-act journey to empower our Hero to conquer this Villain. Each method is a tool, a weapon, or a new sense to perceive and control the world around them.""")
act1, act2, act3 = st.columns(3)
with act1: st.subheader("Act I: Know Thyself (The Foundation)"); st.markdown("Before the battle, the Hero must understand their own strengths and weaknesses. What is the true capability of their measurement system? What are its limits? This is the foundational work of **Characterization and Validation**.")
with act2: st.subheader("Act II: The Transfer (The Crucible)"); st.markdown("The Hero's validated method must now survive in a new land—the receiving QC lab. This is the ultimate test of **Robustness, Stability, and Comparability**. It is here that many battles with Variation are won or lost.")
with act3: st.subheader("Act III: The Guardian (Beyond the Transfer)"); st.markdown("The assay is live, but the Villain never sleeps. The Hero must now become a guardian, using advanced tools to **Monitor, Predict, and Protect** the process for its entire lifecycle, anticipating problems before they arise.")
st.divider()

# --- Sidebar Controls ---
st.sidebar.title("Toolkit Navigation")
st.sidebar.markdown("Select a statistical method to analyze and visualize.")
method_key = st.sidebar.radio("Select a Method:", options=[
    "1. Gage R&R", "2. Linearity and Range", "3. LOD & LOQ", "4. Method Comparison",
    "5. Assay Robustness (DOE/RSM)", "6. Process Stability (Shewhart)", "7. Small Shift Detection",
    "8. Run Validation", "9. Process Capability (Cpk)", "10. Anomaly Detection (ML)",
    "11. Predictive QC (ML)", "12. Control Forecasting (AI)", "13. Pass/Fail Analysis",
    "14. Bayesian Inference", "15. Confidence Interval Concept"
])
st.header(method_key)

# --- Dynamic Content Display ---
# All 15 elif blocks follow, each with the full, detailed content and professional layout.

if "Gage R&R" in method_key:
    st.markdown("""
    **Purpose:** To quantify the inherent variability (error) of the measurement system itself, separating it from the actual process variation. 
    
    **Definition:** A Gage R&R study partitions the total observed variation into two main sources: the variation from the parts being measured and the variation from the measurement system. The measurement system variation is further broken down into **Repeatability** (equipment variation) and **Reproducibility** (operator variation).
    
    **Application:** This is the first and most critical gate in an assay transfer. You cannot validate a process with an unreliable measurement system. Before our Hero can fight the Villain of Process Variation, they must first prove their own weapon—the assay—is sharp, true, and reliable.
    """)
    col1, col2 = st.columns([0.65, 0.35]);
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse")
            st.metric(label="💡 KPI: % Part Variation", value=f"{pct_part:.1f}%", delta="Higher is better")
            st.markdown("- **Run Chart by Part:** Look for consistency. Are the measurements for each part tightly clustered? A wide spread indicates poor repeatability.")
            st.markdown("- **Variation by Operator:** Look for alignment. Do the boxes for each operator overlap significantly? If one operator's box is much higher or lower, it indicates a reproducibility problem (bias).")
            st.markdown("**The Bottom Line:** A low % Gage R&R (<10%) proves that your measurement system is a reliable 'ruler' and that most of the variation you see in your process is real, not just measurement noise.")
        with tab2:
            st.markdown("Based on AIAG (Automotive Industry Action Group) guidelines:")
            st.markdown("- **< 10%:** System is **acceptable**.")
            st.markdown("- **10% - 30%:** **Conditionally acceptable**, depending on the importance of the application and cost of improvement.")
            st.markdown("- **> 30%:** System is **unacceptable** and requires improvement.")
        with tab3:
            st.markdown("**Origin:** Formalized by the AIAG. ANOVA is the preferred method.")
            st.markdown("**Mathematical Basis:** ANOVA partitions total variance ($SS_T$) into components: $SS_T = SS_{Part} + SS_{Operator} + ...$. From this, we derive variance components for repeatability ($\hat{\sigma}^2_{EV}$) and reproducibility ($\hat{\sigma}^2_{AV}$) to calculate:")
            st.latex(r"\%R\&R = 100 \times \left( \frac{\hat{\sigma}_{R\&R}}{\hat{\sigma}_{Total}} \right)")

elif "Linearity and Range" in method_key:
    st.markdown("""
    **Purpose:** To verify the assay's ability to provide results that are directly proportional to the concentration of the analyte across a specified range.
    
    **Definition:** Linearity is the measure of how well a calibration plot of response versus concentration approximates a straight line. The Range is the interval between the upper and lower concentration of an analyte in a sample for which the assay has been demonstrated to have a suitable level of precision, accuracy, and linearity.
    
    **Application:** This study is a fundamental part of assay validation. Our Hero must prove that their weapon—the assay—is not just precise, but also consistently accurate across the entire range of interest. A non-linear assay is like a warped ruler; it may be right in the middle, but it gives dangerously misleading results at the extremes.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig, model = plot_linearity()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}")
            st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}")
            st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}")
            st.markdown("- **Linearity Plot:** Visually confirms the straight-line relationship.")
            st.markdown("- **Residual Plot:** The most powerful diagnostic tool. A random scatter confirms linearity; a curve or funnel shape reveals a problem.")
            st.markdown("- **Recovery Plot:** Directly assesses accuracy at each level. Points falling outside the 80-120% limits indicate a bias at those concentrations.")
            st.markdown("**The Bottom Line:** A high R², a slope near 1, an intercept near 0, random residuals, and recovery within limits provide statistical proof that your assay is trustworthy across its entire reportable range.")
        with tab2:
            st.markdown("- **R² > 0.995** is typically required.")
            st.markdown("- **Slope** should be close to 1.0 (e.g., within **0.95 - 1.05**).")
            st.markdown("- The **95% CI for the Intercept** should contain **0**.")
            st.markdown("- **Recovery** at each level should be within a pre-defined range (e.g., **80% to 120%**).")
        with tab3:
            st.markdown("**Origin:** Based on Ordinary Least Squares (OLS) regression, a fundamental statistical method developed by Legendre and Gauss in the early 1800s.")
            st.markdown("**Mathematical Basis:** We fit the model and test the hypotheses $H_0: \\beta_1 = 1$ and $H_0: \\beta_0 = 0$.")
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("**Recovery:**")
            st.latex(r"\%\,Recovery = \frac{\text{Measured Concentration}}{\text{Nominal Concentration}} \times 100")

elif "LOD & LOQ" in method_key:
    st.markdown("""
    **Purpose:** To determine the lowest concentration of an analyte that the assay can reliably detect (LOD) and accurately quantify (LOQ).
    
    **Definition:**
    - **Limit of Detection (LOD):** The lowest analyte concentration that produces a signal distinguishable from the background noise of the blank. It answers the question, "Is the analyte present?"
    - **Limit of Quantitation (LOQ):** The lowest analyte concentration that can be measured with an acceptable level of precision and accuracy. This is the official lower boundary of the assay's reportable range.
    
    **Application:** This is a critical part of assay characterization. Our Hero must know the limits of their senses. This analysis defines the absolute floor of your assay's capability, proving you can trust measurements down to the LOQ and detect presence down to the LOD. This is vital for applications like impurity testing or low-level biomarker detection.
    """)
    col1, col2 = st.columns([0.65, 0.35]);
    with col1:
        fig, lod_val, loq_val = plot_lod_loq()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} ng/mL")
            st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} ng/mL")
            st.markdown("- **Signal Distribution:** The violin plot (top) visually confirms that the distribution of the low-concentration samples is clearly separated from the distribution of the blank samples.")
            st.markdown("- **Calibration Curve (Low End):** The regression plot (bottom) confirms the assay is linear at the low end of the range. The LOD and LOQ are derived from the variability of the residuals (residual standard error) and the slope of this line.")
            st.markdown("**The Bottom Line:** This analysis defines the absolute floor of your assay's capability. It proves you can trust measurements down to the LOQ, and detect presence down to the LOD.")
        with tab2:
            st.markdown("- The primary acceptance criterion is that the experimentally determined **LOQ must be less than or equal to the lowest concentration that needs to be measured** for the assay's intended use (e.g., the specification limit for an impurity).")
        with tab3:
            st.markdown("**Origin:** Based on the recommendations from the International Council for Harmonisation (ICH) Q2(R1) guidelines.")
            st.markdown("**Mathematical Basis (Calibration Curve Method):** This is the preferred, most robust method. It uses the standard deviation of the residuals (or y-intercepts) from a low-level calibration curve ($\sigma$) and the slope of that curve (S).")
            st.latex(r"LOD = \frac{3.3 \times \sigma}{S}")
            st.latex(r"LOQ = \frac{10 \times \sigma}{S}")

elif "Method Comparison" in method_key:
    st.markdown("""
    **Purpose:** To formally assess the agreement and bias between two different measurement methods (e.g., a new assay vs. a gold standard, or the R&D lab vs. the QC lab).
    
    **Definition:** This analysis quantifies the systematic and random differences between two methods that are intended to measure the same quantity. It goes far beyond a simple correlation to determine if the methods can be used interchangeably.
    
    **Application:** This is the heart of the "Crucible" act in our Hero's journey. Having forged a new weapon (the assay), the Hero must now prove it is as good as the old one. This study provides the definitive evidence to answer the critical question: "Do these two methods agree sufficiently well?" A successful outcome is a cornerstone of a successful assay transfer or method validation.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig, slope, intercept, bias, ua, la = plot_method_comparison()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Mean Bias (B-A)", value=f"{bias:.2f} units")
            st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}", help="Ideal = 1.0. Measures proportional bias.")
            st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}", help="Ideal = 0.0. Measures constant bias.")
            st.markdown("- **Deming Regression:** The confidence band shows the uncertainty. If the 'Line of Identity' falls within this band, there is no significant systematic bias.")
            st.markdown("- **Bland-Altman Plot:** Visualizes the random error and quantifies the expected range of disagreement (Limits of Agreement). Look for trends or funnel shapes, which indicate non-constant bias.")
            st.markdown("- **% Bias Plot:** Directly assesses practical significance. Does the bias at any concentration exceed the pre-defined limits (e.g., ±15%)?")
            st.markdown("**The Bottom Line:** Passing all three analyses proves that the receiving lab's method is statistically indistinguishable from the reference method, confirming a successful transfer.")
        with tab2:
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope should contain 1.0**, and the 95% CI for the **intercept should contain 0**.")
            st.markdown(f"- **Bland-Altman:** Greater than 95% of the data points must fall within the Limits of Agreement (`{la:.2f}` to `{ua:.2f}`). The width of this interval must be practically or clinically acceptable.")
            st.markdown("- **Percent Bias:** The bias at each concentration level should not exceed a pre-defined limit, often **±15%**. ")
        with tab3:
            st.markdown("**Origin:** Deming Regression (W. Edwards Deming) is an errors-in-variables model. The Bland-Altman plot (1986 Lancet paper) was created to properly assess agreement between two clinical measurement methods.")
            st.markdown("**Mathematical Basis (Bland-Altman):**")
            st.markdown("For each sample $i$, calculate Average $(\\frac{x_i+y_i}{2})$ and Difference $(y_i - x_i)$. The Limits of Agreement (LoA) are defined as:")
            st.latex(r"\bar{d} \pm 1.96 \cdot s_d")
            st.markdown("where $\\bar{d}$ and $s_d$ are the mean and standard deviation of the differences.")
            
elif "Assay Robustness (DOE/RSM)" in method_key:
    st.markdown("""
    **Purpose:** To systematically explore how deliberate variations in assay parameters (e.g., temperature, pH) affect the outcome. This is a crucial step in building a deep understanding of the method.
    
    **Application:** This is the Hero's proactive strike against the Villain of Variation. Instead of waiting for problems, we hunt for them. This study identifies which parameters are critical to control tightly (the vital few) and which are insignificant, allowing us to build a robust process that can withstand real-world fluctuations. It ultimately defines a scientifically proven "safe operating space" for the assay.
    """)
    
    vis_type = st.radio(
        "Select Analysis Stage:", 
        ["📊 **Stage 1: Factor Screening (Pareto Plot)**", "📈 **Stage 2: Process Optimization (2D Contour)**", "🧊 **Stage 2: Process Optimization (3D Surface)**"], 
        horizontal=True,
        help="Start with Screening to find key factors, then use Optimization to find the best settings for those factors."
    )
    
    # This function is now enhanced for world-class rendering.
    fig_pareto, fig_contour, fig_surface, effects = plot_robustness_rsm()
    
    col1, col2 = st.columns([0.65, 0.35])

    with col1:
        if "Screening" in vis_type:
            st.plotly_chart(fig_pareto, use_container_width=True)
        elif "2D Contour" in vis_type:
            st.plotly_chart(fig_contour, use_container_width=True)
        else:
            st.plotly_chart(fig_surface, use_container_width=True)
            
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Most Significant Factor", value=f"{effects.index[0]}")
            st.metric(label="💡 Effect Magnitude", value=f"{effects.values[0]:.2f}")
            st.markdown("- **Screening (Pareto):** The Pareto plot instantly reveals the 'vital few' parameters with significant effects (those colored red). In this case, `Temp` and the `Temp:pH` interaction are the most critical drivers of variation.")
            st.markdown("- **Optimization (Contour/Surface):** These plots provide a map of the process, revealing the 'sweet spot'—the combination of settings that yields the optimal response (highest point on the surface).")
            st.markdown("**The Bottom Line:** This study provides a map of your assay's operating space, allowing you to set control limits that guarantee robustness against real-world process noise.")
        with tab2:
            st.markdown("- **Screening:** Any factor whose effect bar crosses the significance threshold is considered a **critical parameter**. The acceptance rule is that the final SOP must include tighter controls for these parameters.")
            st.markdown("- **Optimization:** The goal is to define a **Design Space** or **Normal Operating Range (NOR)**—a region on the contour plot where the assay is proven to be robust and reliable. The final process parameters should be set well within this space, far from any steep 'cliffs'.")
        with tab3:
            st.markdown("**Origin:** Design of Experiments (DOE) was pioneered by Sir R.A. Fisher. Response Surface Methodology (RSM) was developed by Box and Wilson to efficiently model and optimize processes.")
            st.markdown("**Mathematical Basis:** RSM fits a second-order (quadratic) model to the experimental data, which can capture curvature in the response:")
            st.latex("y = \\beta_0 + \\sum \\beta_i x_i + \\sum \\beta_{ii} x_i^2 + \\sum \\beta_{ij} x_i x_j + \\epsilon")
            
# Replace the ENTIRE 'elif "Process Stability" in method_key:' block with this one.

elif "Process Stability" in method_key:
    st.markdown("""
    **Purpose:** To establish if a process is in a state of statistical control, meaning its variation is stable, consistent, and predictable over time.
    
    **Definition:** A process is "in control" when its variation is due only to random, inherent "common causes." An "out of control" process exhibits "special cause" variation from specific, identifiable events.
    
    **Application:** This is the foundational step of process monitoring and a **strict prerequisite for Process Capability analysis**. It is the Hero's first battle: to prove that the Villain of Variation has been tamed and is behaving predictably. An out-of-control process is a wild, untamed beast; a capable analysis cannot be performed until it is brought into a state of control.
    """)
    col1, col2 = st.columns([0.65, 0.35]);
    with col1:
        st.plotly_chart(plot_shewhart(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Process Stability", value="Signal Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **I-Chart (top):** Monitors the process center (accuracy). The single blue line shows the continuous process. Points marked with a red 'X' are out-of-control signals.")
            st.markdown("- **MR-Chart (bottom):** Monitors the short-term, run-to-run variability (precision). An out-of-control signal here would indicate the process has become inconsistent.")
            st.markdown("**The Bottom Line:** These charts are the heartbeat of your process. This chart shows a stable heartbeat for the first 15 runs, after which a new reagent lot caused a special cause variation, driving the process out of control. This must be fixed before proceeding.")
        with tab2:
            st.markdown("- A process is considered stable and ready for the next validation step only when **at least 20-25 consecutive points on both the I-chart and MR-chart show no out-of-control signals** according to the chosen rule set (e.g., Nelson, Westgard).")
        with tab3:
            st.markdown("**Origin:** Developed by Walter A. Shewhart at Bell Labs in the 1920s, these charts are the foundation of modern Statistical Process Control (SPC).")
            st.markdown("**Mathematical Basis:** The key is estimating the process standard deviation ($\hat{\sigma}$) from the average moving range ($\overline{MR}$).")
            st.latex(r"\hat{\sigma} = \frac{\overline{MR}}{d_2}")
            st.markdown("Where $d_2$ is a control chart constant (1.128 for a moving range of size 2).")
            st.markdown("**I-Chart Limits:**")
            st.latex(r"UCL/LCL = \bar{x} \pm 3\hat{\sigma}")
            st.markdown("**MR-Chart Limits:**")
            st.latex(r"UCL = D_4 \overline{MR}")
            st.markdown("Where $D_4$ is another constant (3.267 for a moving range of size 2).")
            
elif "Small Shift Detection" in method_key:
    st.markdown("""
    **Purpose:** To implement sensitive charts that can detect small, sustained shifts in the process mean that a standard Shewhart chart would miss.
    
    **Definition:**
    - **EWMA (Exponentially Weighted Moving Average):** A chart that gives exponentially decreasing weight to older observations, making it sensitive to small, gradual drifts.
    - **CUSUM (Cumulative Sum):** A chart that accumulates deviations from a target value, making it the most statistically powerful tool for detecting small, abrupt, and sustained shifts.
    
    **Application:** These are the Hero's early-warning systems. After the initial chaos of transfer is tamed (Act II), the Villain of Variation becomes subtle. It introduces slow instrument drift or a slight bias from a new reagent lot. A Shewhart chart might not notice, but these advanced charts will, allowing the Hero to act *before* a major out-of-spec event occurs.
    """)
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM'))
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        if chart_type == 'EWMA':
            lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05, help="Smaller λ is more sensitive to small shifts but slower to react.")
            st.plotly_chart(plot_ewma_cusum(chart_type, lmbda, 0, 0), use_container_width=True)
        else:
            k_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25, help="Set to half the size of the shift you want to detect (e.g., 0.5 to detect a 1-sigma shift).")
            H_sigma = st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5, help="The decision interval or control limit.")
            st.plotly_chart(plot_ewma_cusum(chart_type, 0, k_sigma, H_sigma), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Shift Detection", value="Signal Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **Top Plot (Raw Data):** Notice how the small 1.25σ shift after run 25 is almost impossible to see by eye.")
            st.markdown("- **Bottom Plot (EWMA/CUSUM):** This chart makes the invisible visible. It accumulates the small deviations until they cross the red control limit, providing a clear statistical signal.")
            st.markdown("**The Bottom Line:** These charts act as a magnifying glass for the process mean, allowing you to catch subtle problems early and maintain a high level of quality and consistency.")
        with tab2:
            st.markdown("- **EWMA Rule:** For long-term monitoring, a `λ` between **0.1 to 0.3** is a common choice. A signal occurs when the EWMA line crosses the control limits.")
            st.markdown("- **CUSUM Rule:** To detect a shift of size $\delta$, set the slack parameter `k` to approximately **$\delta / 2$**. A signal occurs when the CUSUM statistic crosses the decision interval `H`.")
        with tab3:
            st.markdown("**Origin:** EWMA (Roberts, 1959); CUSUM (Page, 1954). These were developed to overcome the Shewhart chart's relative insensitivity to small, sustained shifts.")
            st.markdown("**Mathematical Basis (CUSUM):** Two statistics accumulate deviations above and below the target, incorporating a slack value, k:")
            st.latex("SH_i = \\max(0, SH_{i-1} + (x_i - \\mu_0) - k)")
            st.latex("SL_i = \\max(0, SL_{i-1} + (\\mu_0 - x_i) - k)")
            st.markdown("A signal occurs if $SH_i$ or $SL_i > H$.")

elif "Run Validation" in method_key:
    st.markdown("""
    **Purpose:** To create an objective, statistically-driven system for accepting or rejecting each individual analytical run based on the performance of Quality Control (QC) samples.
    
    **Definition:** Run validation employs a set of statistical rules, applied to a Levey-Jennings chart, to detect deviations from expected performance. These rules are designed to catch both large, random errors and smaller, systematic trends that might indicate a problem.
    
    **Application:** This is the Hero's daily duty as the Guardian of the process. Each day, the Villain of Variation will try to sneak past the defenses. This multi-rule system is the vigilant gatekeeper that ensures only valid, trustworthy results are released. It is the core of routine QC in any regulated lab environment.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.plotly_chart(plot_multi_rule(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Run Status", value="Violations Detected", delta="Action Required", delta_color="inverse")
            st.markdown("- **Levey-Jennings Chart:** The top plot visualizes the QC data over time with 1, 2, and 3-sigma zones. Specific Westgard rule violations are now automatically flagged and annotated.")
            st.markdown("- **Distribution Plot:** The bottom plot shows the overall histogram of the QC data. It should approximate a bell curve centered on the mean. Skewness or multiple peaks can indicate a problem.")
            st.markdown("**The Bottom Line:** The annotations on the Levey-Jennings chart provide immediate, actionable feedback. They distinguish between random errors (like the `R_4s` rule) and systematic errors (like the `2_2s` or `4_1s` rules), guiding the troubleshooting process.")
        with tab2:
            st.markdown("A run is typically rejected if a 'Rejection Rule' is violated. See the tabs below for detailed rule sets.")
        with tab3:
            st.markdown("**Origin:** Dr. S. Levey and Dr. E. R. Jennings introduced control charts to the clinical lab in the 1950s. Dr. James Westgard later developed the multi-rule system in the 1980s to improve error detection.")
            st.markdown("**Mathematical Basis:** The rules are based on the probabilities of points from a normal distribution falling within specific sigma zones. Patterns that are highly improbable under normal conditions are flagged as signals.")
            
    st.subheader("Standard Industry Rule Sets")
    tab_w, tab_n, tab_we = st.tabs(["✅ Westgard Rules", "✅ Nelson Rules", "✅ Western Electric Rules"])
    with tab_w: st.markdown("""Developed for lab QC, vital for CLIA, CAP, ISO 15189 compliance. A run is rejected if a "Rejection Rule" is violated.
| Rule | Use Case | Interpretation |
|---|---|---|
| **1_2s** | Warning | One control > ±2σ. Triggers inspection. |
| **1_3s** | Rejection | One control > ±3σ. |
| **2_2s** | Rejection | Two consecutive > same ±2σ limit. |
| **R_4s** | Rejection | One > +2σ and the next > -2σ. |
| **4_1s** | Rejection | Four consecutive > same ±1σ limit. |
| **10x** | Rejection | Ten consecutive points on the same side of the mean. |""")
    with tab_n: st.markdown("""Excellent for catching non-random patterns in manufacturing and general SPC.
| Rule | What It Flags |
|---|---|
| 1. One point > 3σ | Sudden shift or outlier |
| 2. 9 points on same side of mean | Mean shift |
| 3. 6 points increasing or decreasing | Trend |
| 4. 14 points alternating up/down | Systematic oscillation |
| 5. 2 of 3 > 2σ (same side) | Moderate shift |
| 6. 4 of 5 > 1σ (same side) | Small persistent shift |
| 7. 15 points inside ±1σ | Reduced variation |
| 8. 8 points outside ±1σ | Increased variation |""")
    with tab_we: st.markdown("""Foundational rules from which many other systems were derived.
| Rule | Interpretation |
|---|---|
| **Rule 1** | One point falls outside the ±3σ limits. |
| **Rule 2** | Two out of three consecutive points fall beyond the ±2σ limit on the same side. |
| **Rule 3** | Four out of five consecutive points fall beyond the ±1σ limit on the same side. |
| **Rule 4** | Eight consecutive points fall on the same side of the mean. |""")

elif "Process Capability" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To determine if the stable process is capable of consistently producing results that meet specifications. **Application:** This is often the final gate of a transfer, proving the new site can meet quality targets.")
    scenario = st.sidebar.radio("Select Process Scenario:", ('Ideal', 'Shifted', 'Variable', 'Out of Control'))
    col1, col2 = st.columns([0.65, 0.35])
    with col1: fig, cpk_val, scn = plot_capability(scenario); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scn != 'Out of Control' else "INVALID")
            st.markdown("- **The Mantra:** Control before Capability. Cpk is only meaningful for a stable, in-control process (see I-Chart in the plot).")
            st.markdown("- **The 'Holy Shit' Moment:** A process can be perfectly **in control but not capable** (the 'Shifted' and 'Variable' scenarios). The control chart looks fine, but the process is producing scrap. This is why you need both tools.")
        with tab2:
            st.markdown("- `Cpk ≥ 1.33`: Process is **capable**."); st.markdown("- `Cpk ≥ 1.67`: Process is **highly capable**."); st.markdown("- `Cpk < 1.0`: Process is **not capable**.")
        with tab3:
            st.markdown("**Origin:** Developed in manufacturing as part of Six Sigma."); st.markdown("**Mathematical Basis:** $ C_{pk} = \\min \\left( \\frac{USL - \\bar{x}}{3\\hat{\sigma}}, \\frac{\\bar{x} - LSL}{3\\hat{\sigma}} \\right) $.")

elif "Anomaly Detection (ML)" in method_key:
    st.markdown("""
    **Purpose:** To leverage machine learning to detect complex, multivariate anomalies that traditional univariate control charts would miss.
    
    **Definition:** An anomaly is a data point that deviates significantly from the majority of the data. An Isolation Forest is an unsupervised algorithm that identifies these points by learning the 'shape' of normal operating conditions.
    
    **Application:** This is the Hero's tool for finding the "ghost in the machine." An operator might swear every individual parameter is in spec, but the ML model flags a run as an anomaly because the *combination* of parameters is highly unusual. This is critical for uncovering subtle, novel failure modes that would otherwise go unnoticed.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Anomalies Detected", value="3", help="Number of points flagged by the model.")
            st.markdown("- **The Plot:** The blue shaded area represents the model's learned 'normal' operating space. Points outside this area are flagged as anomalies (red).")
            st.markdown("- **The 'Holy Shit' Moment:** Notice that the anomalous points are not necessarily extreme on any single axis. Their combination is what makes them unusual, a pattern that is nearly impossible for a human or a simple control chart to detect.")
            st.markdown("**The Bottom Line:** This is a proactive monitoring tool that moves beyond simple rule-based alarms to a holistic assessment of process health, enabling the detection of previously unknown problems.")
        with tab2:
            st.markdown("- This is an exploratory and monitoring tool. There is no hard 'pass/fail' rule during validation.")
            st.markdown("- The primary rule is that any point flagged as an **anomaly must be investigated** by Subject Matter Experts (SMEs) to determine the root cause and assess its impact on product quality.")
        with tab3:
            st.markdown("**Origin:** The Isolation Forest algorithm was proposed by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in 2008.")
            st.markdown("**Mathematical Basis:** It is based on the principle that anomalies are 'few and different' and thus easier to isolate in a random tree structure. The model's anomaly score is based on the average path length required to isolate a data point.")
            st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")

elif "Predictive QC (ML)" in method_key:
    st.markdown("""
    **Purpose:** To move from *reactive* quality control (detecting a failure after it happens) to *proactive* failure prevention.
    
    **Definition:** A predictive QC model is a machine learning classifier (like Logistic Regression) that learns the relationship between in-process parameters and the final outcome of a run (Pass/Fail).
    
    **Application:** This is the Hero's crystal ball. Before committing expensive reagents and valuable time, the model can look at the starting conditions of a run (e.g., reagent age, instrument warmup time) and predict its likelihood of success. A high probability of failure can trigger an alert, empowering the operator to take corrective action *before* the run is wasted. This directly reduces waste and improves right-first-time rates.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig = plot_predictive_qc()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Predictive Risk Profiling", value="Enabled")
            st.markdown("- **Decision Boundary Plot (left):** This is the model's 'risk map.' The color gradient shows the predicted probability of failure, from low (green) to high (red). The overlaid points show how well the model classified the historical data.")
            st.markdown("- **Probability Distribution Plot (right):** This is the model's report card. It shows the predicted failure probabilities for runs that actually passed (green distribution) versus runs that actually failed (red distribution).")
            st.markdown("- **The 'Holy Shit' Moment:** A great model shows a clear separation between the green and red distributions. This proves that the model has learned the hidden patterns that lead to failure and can reliably distinguish a good run from a bad one before it's too late.")
            st.markdown("**The Bottom Line:** This tool transforms quality control from a pass/fail-at-the-end activity to an in-process, risk-based decision-making tool.")
        with tab2:
            st.markdown("- A risk threshold is established based on the model and business needs. For example, 'If the model's predicted **Probability of Failure is > 20%**, flag the run for mandatory operator review before proceeding.'")
            st.markdown("- The model's performance (e.g., accuracy, sensitivity) must be formally validated and documented before use in a regulated environment.")
        with tab3:
            st.markdown("**Origin:** Logistic regression is a statistical model developed by statistician David Cox in 1958. It is a foundational and highly interpretable algorithm for binary classification problems.")
            st.markdown("**Mathematical Basis:** It models the probability of a binary outcome (y=1) by passing a linear combination of input features ($x$) through the sigmoid (logistic) function:")
            st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ...)}}")

elif "Control Forecasting (AI)" in method_key:
    st.markdown("""
    **Purpose:** To use a time series model to forecast the future performance of assay controls, enabling proactive management instead of reactive problem-solving.
    
    **Definition:** Time series forecasting is a machine learning technique that analyzes historical, time-ordered data points to detect patterns (like trend and seasonality) and uses them to predict future values.
    
    **Application:** This is the Hero's ultimate power: seeing the future. By forecasting where a control is heading, the Hero can anticipate problems *before* they occur. This tool can predict that a reagent lot will start to fail in 3 weeks, or that an instrument will need recalibration next month. It transforms maintenance and inventory management from a scheduled chore into an intelligent, data-driven strategy.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig1_fc, fig2_fc, fig3_fc = plot_forecasting()
        st.plotly_chart(fig1_fc, use_container_width=True)
        st.plotly_chart(fig2_fc, use_container_width=True)
        st.plotly_chart(fig3_fc, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Forecast Status", value="Future Breach Predicted", help="The model predicts the spec limit will be crossed.")
            st.markdown("- **Forecast Plot (Top):** Shows the historical data (black dots), the model's prediction (blue line), and the 80% confidence interval (blue band). Red diamonds mark where the forecast breaches the specification limit.")
            st.markdown("- **Trend & Changepoints (Middle):** This is the most powerful diagnostic plot. It shows the underlying long-term trend of the process. Red dashed lines mark 'changepoints' where the model detected a significant shift in the trend, often corresponding to real-world events like a new instrument or reagent lot.")
            st.markdown("- **Seasonality (Bottom):** Shows the predictable yearly pattern in the assay's performance.")
            st.markdown("**The Bottom Line:** This analysis provides a roadmap for the future. It not only tells you *when* a problem is likely to occur but also *why* (is it a long-term trend or a seasonal effect?), enabling highly targeted and proactive process management.")
        with tab2:
            st.markdown("- A **'Proactive Alert'** should be triggered if the lower or upper bound of the 80% forecast interval (`yhat_lower` or `yhat_upper`) is predicted to cross a specification limit within the defined forecast horizon (e.g., the next 4-6 weeks).")
            st.markdown("- Any automatically detected **changepoint** should be investigated and correlated with historical batch records or lab events to understand its root cause.")
        with tab3:
            st.markdown("**Origin:** Prophet is an open-source forecasting procedure developed by Facebook's Core Data Science team, designed to be robust for business-style time series data.")
            st.markdown("**Mathematical Basis:** It's a decomposable time series model that fits a curve using a combination of trend, seasonality, and holiday components.")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("- $g(t)$: Piecewise linear or logistic growth trend with automated changepoint detection.")
            st.markdown("- $s(t)$: Seasonal patterns (e.g., yearly, weekly) modeled with Fourier series.")
            st.markdown("- $h(t)$: Effects of known, potentially irregular events (holidays).")

elif "Pass/Fail Analysis" in method_key:
    st.markdown("""
    **Purpose:** To accurately calculate and compare confidence intervals for a binomial proportion.
    
    **Definition:** A binomial proportion is the ratio of successes to the total number of trials (e.g., number of concordant results / total samples). A confidence interval provides a range of plausible values for the true, underlying proportion.
    
    **Application:** This is essential for validating qualitative or "pass/fail" assays. Our Hero needs to prove, with a high degree of confidence, that the assay's success rate (e.g., concordance with a reference method) is above a certain threshold. Choosing the wrong statistical method here can lead to dangerously misleading conclusions, especially with the small sample sizes common in validation studies.
    """)
    n_samples_wilson = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30, key='wilson_n')
    successes_wilson = st.sidebar.slider("Concordant Results", 0, n_samples_wilson, int(n_samples_wilson * 0.95), key='wilson_s')
    
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig1_wilson, fig2_wilson = plot_wilson(successes_wilson, n_samples_wilson)
        st.plotly_chart(fig1_wilson, use_container_width=True)
        st.plotly_chart(fig2_wilson, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}")
            st.markdown("- **CI Comparison (Top Plot):** This plot shows the calculated 95% confidence intervals from three different methods. The error bars represent the uncertainty. Notice how the unreliable 'Wald' interval can become dangerously narrow or extend beyond plausible values (0 or 1).")
            st.markdown("- **Coverage Probability (Bottom Plot):** This is the model's report card. It shows the *actual* probability that the interval contains the true value for a given true proportion. An ideal interval would be a flat line at 95%.")
            st.markdown("- **The 'Holy Shit' Moment:** The Wald interval's coverage (in red) is terrible. It frequently drops to dangerously low levels, meaning it gives a false sense of precision. The Wilson and Clopper-Pearson intervals perform much closer to the nominal 95% level, proving their reliability.")
            st.markdown("**The Bottom Line:** Never use the standard Wald interval for important decisions. The Wilson Score interval provides the best balance of accuracy and interval width for most applications.")
        with tab2:
            st.markdown("- A common acceptance criterion for assay validation is: **'The lower bound of the 95% Wilson Score (or Clopper-Pearson) confidence interval must be greater than or equal to the target concordance rate'** (e.g., 90%).")
        with tab3:
            st.markdown("**Origin:** The Wilson Score (1927) and Clopper-Pearson (1934) intervals were developed to provide much better performance than the standard Wald interval, especially for small samples.")
            st.markdown("**Mathematical Basis (Wilson):**")
            st.latex(r"\frac{1}{1 + z^2/n} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right)")
            
elif "Bayesian Inference" in method_key:
    st.markdown("""
    **Purpose:** To formally combine existing knowledge (the 'Prior') with new experimental data (the 'Likelihood') to arrive at an updated, more robust conclusion (the 'Posterior').
    
    **Definition:** Bayesian inference is a statistical framework that treats parameters not as fixed unknown constants, but as random variables about which we can have a belief that is updated as we gather new evidence.
    
    **Application:** This is the Hero's secret weapon for efficiency. Instead of starting from scratch, the Hero can leverage the vast knowledge from the R&D lab (the Prior) to design smaller, smarter validation studies at the QC site (the Likelihood). It answers the question: "Given what we already knew, what does this new data tell us now?"
    """)
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig, prior_mean, mle, posterior_mean = plot_bayesian(prior_type_bayes)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}", help="The final, data-informed belief.")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}", help="The initial belief before seeing the new data.")
            st.metric(label="💡 Data-only Estimate (MLE)", value=f"{mle:.3f}", help="The evidence from the new data alone.")
            st.markdown("- **Prior (Green):** Our initial belief. A 'Strong' prior is narrow and confident; a 'Skeptical' prior is broad and uncertain.")
            st.markdown("- **Likelihood (Red):** The 'voice of the data'—the hard evidence from the new experiment.")
            st.markdown("- **Posterior (Blue):** The final, updated belief. It's a weighted compromise, pulled from the Prior towards the Likelihood.")
            st.markdown("- **The 'Holy Shit' Moment:** Switch the sidebar from 'Strong R&D Prior' to 'Skeptical Prior'. Watch how the blue Posterior dramatically shifts. With a strong prior, our belief barely moves despite the new data. With a skeptical prior, the new data almost completely dictates our final belief. This is Bayesian updating in action.")
        with tab2:
            st.markdown("- The **95% credible interval must be entirely above the target** (e.g., 90%).")
            st.markdown("- This approach allows for demonstrating success with smaller sample sizes if a strong, justifiable prior is used.")
        with tab3:
            st.markdown("**Origin:** Based on Bayes' Theorem (18th century), but made practical by modern computational methods.")
            st.markdown("**Mathematical Basis:** The core idea is that the posterior is proportional to the product of the likelihood and the prior.")
            st.latex(r"\text{Posterior} \propto \text{Likelihood} \times \text{Prior}")
            st.markdown("For binomial data, we use the Beta-Binomial conjugate model: if the Prior is Beta($\\alpha, \\beta$) and we observe $k$ successes in $n$ trials, the Posterior is Beta($\\alpha + k, \\beta + n - k$).")
            

elif "Confidence Interval Concept" in method_key:
    st.markdown("""
    **Purpose:** To understand the fundamental concept of frequentist confidence intervals and how sample size directly impacts the precision of our estimates.
    
    **Definition:** A confidence interval (CI) is a range of estimates for an unknown population parameter, calculated from sample data. Its width reflects the uncertainty of the estimate.
    
    **Application:** This is a foundational concept that underpins many of the statistical tests used in validation and quality control. For the Hero, understanding this is not just academic; it's about resource management. How many samples do I need to run to be confident in my result? This interactive simulation provides a powerful, actionable answer by letting you see the direct trade-off between sample size, cost, and statistical precision.
    """)
    n_slider = st.sidebar.slider("Select Sample Size (n) for Simulation:", 5, 100, 30, 5)
    
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig1_ci, fig2_ci, capture_count, n_sims, avg_width = plot_ci_concept(n=n_slider)
        st.plotly_chart(fig1_ci, use_container_width=True)
        st.plotly_chart(fig2_ci, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ The Golden Rule", "📖 Method Theory"])
        with tab1:
            st.metric(label=f"📈 KPI: Average CI Width (n={n_slider})", value=f"{avg_width:.2f} units")
            st.metric(label="💡 Empirical Coverage", value=f"{(capture_count/n_sims):.0%}", help="The % of simulated CIs that captured the true mean.")
            st.markdown("- **Sampling Distribution (Top Plot):** As you increase the sample size `n` with the slider, watch the orange curve (the distribution of sample means) become narrower. This is the Central Limit Theorem in action!")
            st.markdown("- **CI Simulation (Bottom Plot):** As `n` increases, the confidence intervals (the blue and red lines) become dramatically shorter. This is the payoff for collecting more data: a more precise estimate.")
            st.markdown("- **The 'Holy Shit' Moment (Diminishing Returns):** Notice that the gain in precision from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This demonstrates the law of diminishing returns in sampling, a critical concept for designing efficient experiments.")
        with tab2:
            st.markdown("This is a teaching module, not a validation step. The 'acceptance rule' is to **correctly interpret the CI** in all reports and discussions:")
            st.error("🔴 **Incorrect:** 'There is a 95% probability that the true mean is in this interval.'")
            st.success("🟢 **Correct:** 'We are 95% confident that this interval contains the true mean.' This is shorthand for: 'This interval was constructed using a procedure that, in the long run, captures the true mean 95% of the time.'")
        with tab3:
            st.markdown("**Origin:** The concept of confidence intervals was introduced by Polish mathematician Jerzy Neyman in the 1930s.")
            st.markdown("**Mathematical Basis:** The general form is:")
            st.latex(r"\text{CI} = \text{Point Estimate} \pm (\text{Critical Value}) \times (\text{Standard Error})")
            st.markdown("For the mean with an unknown population standard deviation (the most common case), this becomes:")
            st.latex(r"\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}")
