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
    np.random.seed(42); nominal = np.array([10, 25, 50, 100, 150, 200, 250]); measured = nominal + np.random.normal(0, 2, len(nominal)) - (nominal/150)**3; X = sm.add_constant(nominal); model = sm.OLS(measured, X).fit(); b, m = model.params
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linearity Plot", "Residual Analysis")); fig.add_trace(go.Scatter(x=nominal, y=measured, mode='markers', name='Measured Values'), row=1, col=1); fig.add_trace(go.Scatter(x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line'), row=1, col=1); fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1); fig.add_trace(go.Scatter(x=nominal, y=model.resid, mode='markers', name='Residuals', marker_color='green'), row=1, col=2); fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2); fig.update_layout(title_text='Assay Linearity and Range Verification', showlegend=True, height=600); fig.update_xaxes(title_text="Nominal Concentration (ng/mL)", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration (ng/mL)", row=1, col=1); fig.update_xaxes(title_text="Nominal Concentration (ng/mL)", row=1, col=2); fig.update_yaxes(title_text="Residuals (Measured - Predicted)", row=1, col=2); return fig, model

def plot_lod_loq():
    np.random.seed(3); blanks = np.random.normal(1.5, 0.5, 20); low_conc = np.random.normal(5.0, 0.6, 20); mean_blank, std_blank = np.mean(blanks), np.std(blanks, ddof=1); LOD = mean_blank + 3.3 * std_blank; LOQ = mean_blank + 10 * std_blank; x_kde = np.linspace(0, 8, 200); kde_blanks = stats.gaussian_kde(blanks)(x_kde); kde_low = stats.gaussian_kde(low_conc)(x_kde)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=x_kde, y=kde_blanks, fill='tozeroy', name='Blank Sample Distribution')); fig.add_trace(go.Scatter(x=x_kde, y=kde_low, fill='tozeroy', name='Low Conc. Sample Distribution')); fig.add_vline(x=LOD, line_dash="dash", line_color="orange", annotation_text=f"LOD={LOD:.2f}"); fig.add_vline(x=LOQ, line_dash="dash", line_color="red", annotation_text=f"LOQ={LOQ:.2f}"); fig.update_layout(title_text='Limit of Detection (LOD) and Quantitation (LOQ)', xaxis_title='Assay Signal (e.g., Absorbance)', yaxis_title='Density', height=600); return fig, LOD, LOQ

def plot_method_comparison():
    np.random.seed(42); x = np.linspace(20, 150, 50); y = 0.98 * x + 1.5 + np.random.normal(0, 2.5, 50); delta = np.var(y, ddof=1) / np.var(x, ddof=1); x_mean, y_mean = np.mean(x), np.mean(y); Sxx = np.sum((x-x_mean)**2); Sxy = np.sum((x-x_mean)*(y-y_mean)); beta1_deming = (np.sum((y-y_mean)**2) - delta*Sxx + np.sqrt((np.sum((y-y_mean)**2) - delta*Sxx)**2 + 4*delta*Sxy**2)) / (2*Sxy); beta0_deming = y_mean - beta1_deming*x_mean; diff = y - x; mean_diff = np.mean(diff); upper_loa = mean_diff + 1.96*np.std(diff,ddof=1); lower_loa = mean_diff - 1.96*np.std(diff,ddof=1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Deming Regression", "Bland-Altman Agreement Plot")); fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Sample Results'), row=1, col=1); fig.add_trace(go.Scatter(x=x, y=beta0_deming + beta1_deming*x, mode='lines', name='Deming Fit'), row=1, col=1); fig.add_trace(go.Scatter(x=[0, 160], y=[0, 160], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1); fig.add_trace(go.Scatter(x=(x+y)/2, y=diff, mode='markers', name='Difference', marker_color='purple'), row=1, col=2); fig.add_hline(y=mean_diff, line_color="red", annotation_text=f"Mean Bias={mean_diff:.2f}", row=1, col=2); fig.add_hline(y=upper_loa, line_dash="dash", line_color="blue", annotation_text=f"Upper LoA={upper_loa:.2f}", row=1, col=2); fig.add_hline(y=lower_loa, line_dash="dash", line_color="blue", annotation_text=f"Lower LoA={lower_loa:.2f}", row=1, col=2); fig.update_layout(title_text='Method Comparison: R&D Lab vs QC Lab', showlegend=True, height=600); fig.update_xaxes(title_text="R&D Lab (Reference)", row=1, col=1); fig.update_yaxes(title_text="QC Lab (Test)", row=1, col=1); fig.update_xaxes(title_text="Average of Methods", row=1, col=2); fig.update_yaxes(title_text="Difference (QC - R&D)", row=1, col=2); return fig, beta1_deming, beta0_deming, mean_diff, upper_loa, lower_loa

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
    np.random.seed(101); data = np.concatenate([np.random.normal(50, 2, 25), np.random.normal(52.5, 2, 15)]); target = np.mean(data[:25]); sigma = np.std(data[:25], ddof=1); x_axis = np.arange(1, len(data)+1); fig = go.Figure()
    if chart_type == 'EWMA':
        ewma_vals = np.zeros_like(data); ewma_vals[0] = target;
        for i in range(1, len(data)): ewma_vals[i] = lmbda * data[i] + (1 - lmbda) * ewma_vals[i-1]
        L = 3; UCL = [target + L*sigma*np.sqrt((lmbda/(2-lmbda))*(1-(1-lmbda)**(2*i))) for i in range(1, len(data)+1)]; out_idx = np.where(ewma_vals > UCL)[0]
        fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', name='Daily Control', marker=dict(color='grey', opacity=0.5))); fig.add_trace(go.Scatter(x=x_axis, y=ewma_vals, mode='lines+markers', name=f'EWMA (λ={lmbda})', line=dict(color='purple'))); fig.add_trace(go.Scatter(x=x_axis, y=UCL, mode='lines', name='EWMA UCL', line=dict(color='red'))); fig.update_layout(title_text='EWMA Chart for Detecting Slow Drift', yaxis_title='Assay Response (EU/mL)')
        if len(out_idx) > 0: fig.add_trace(go.Scatter(x=[x_axis[out_idx[0]]], y=[ewma_vals[out_idx[0]]], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Signal'))
    else:
        k = k_sigma * sigma; H = H_sigma * sigma; SH, SL = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)): SH[i] = max(0, SH[i-1] + (data[i] - target) - k); SL[i] = max(0, SL[i-1] + (target - data[i]) - k)
        out_idx = np.where((SH > H) | (SL > H))[0]
        fig.add_trace(go.Scatter(x=x_axis, y=SH, mode='lines+markers', name='High-Side CUSUM (SH)', line=dict(color='darkcyan'))); fig.add_trace(go.Scatter(x=x_axis, y=SL, mode='lines+markers', name='Low-Side CUSUM (SL)', line=dict(color='darkorange'))); fig.add_hline(y=H, line_dash="dash", line_color="red", annotation_text=f"Limit H={H:.1f}"); fig.update_layout(title_text='CUSUM Chart for Detecting Sustained Shifts', yaxis_title='Cumulative Sum');
        if len(out_idx) > 0: fig.add_trace(go.Scatter(x=[x_axis[out_idx[0]]], y=[SH[out_idx[0]]], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Signal'))
    fig.add_vrect(x0=25.5, x1=40.5, fillcolor="orange", opacity=0.2, layer="below", line_width=0, name="Process Shift"); fig.add_hline(y=target if chart_type=='EWMA' else 0, line_dash="dot", line_color="black", annotation_text="Target"); fig.update_layout(height=600, xaxis_title='Analytical Run Number'); return fig

def plot_multi_rule():
    np.random.seed(3); mean, std = 100, 2; data = np.concatenate([np.random.normal(mean, std, 5), [mean + 2.1*std, mean + 2.2*std], np.random.normal(mean, std, 2), np.linspace(mean-0.5*std, mean-2*std, 6), [mean + 3.5*std], np.random.normal(mean + 1.5*std, 0.3, 4), np.random.normal(mean, std, 3), np.random.normal(mean - 1.5*std, 0.3, 5)]); x = np.arange(1, len(data) + 1); fig = go.Figure(); fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers', name='QC Sample', line=dict(color='darkblue')));
    for i, color in zip([1, 2, 3], ['gold', 'orange', 'red']):
        fig.add_hrect(y0=mean+i*std, y1=mean+(i+1)*std, fillcolor=color, opacity=0.1, layer="below", line_width=0); fig.add_hrect(y0=mean-i*std, y1=mean-(i+1)*std, fillcolor=color, opacity=0.1, layer="below", line_width=0)
        fig.add_hline(y=mean+i*std, line_dash="dot", line_color="gray", annotation_text=f"+{i}σ"); fig.add_hline(y=mean-i*std, line_dash="dot", line_color="gray", annotation_text=f"-{i}σ")
    fig.add_hline(y=mean, line_dash="dash", line_color="black", annotation_text="Mean"); fig.update_layout(title_text='QC Run Validation Chart (Levey-Jennings)', xaxis_title='QC Run Number', yaxis_title='Measured Value', height=600); return fig

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
    np.random.seed(42); X_normal = np.random.multivariate_normal([100, 20], [[5, 2],[2, 1]], 200); X_anomalies = np.array([[95, 25], [110, 18], [115, 28]]); X = np.vstack([X_normal, X_anomalies]); model = IsolationForest(n_estimators=100, contamination=0.015, random_state=42); model.fit(X); y_pred = model.predict(X); xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-5, X[:, 0].max()+5, 100), np.linspace(X[:, 1].min()-5, X[:, 1].max()+5, 100)); Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure(); fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, colorscale='Blues_r', showscale=False, opacity=0.4)); fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y_pred, colorscale='coolwarm_r', line=dict(width=1, color='black')), text=[f"Status: {'Anomaly' if p==-1 else 'Normal'}" for p in y_pred], hoverinfo='x+y+text')); fig.update_layout(title_text='Multivariate Anomaly Detection (Isolation Forest)', xaxis_title='Assay Response (Fluorescence Units)', yaxis_title='Incubation Time (min)', height=600); return fig

def plot_predictive_qc():
    np.random.seed(1); n_points = 150; X1 = np.random.normal(5, 2, n_points); X2 = np.random.normal(25, 3, n_points); logit_p = -15 + 1.5 * X1 + 0.5 * X2 + np.random.normal(0, 2, n_points); p = 1 / (1 + np.exp(-logit_p)); y = np.random.binomial(1, p); X = np.vstack([X1, X2]).T; model = LogisticRegression().fit(X, y); xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)); Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    fig = go.Figure(); fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, colorscale='RdYlGn_r', showscale=True, name='P(Fail)')); df_plot = pd.DataFrame({'X1': X[:,0], 'X2': X[:,1], 'Outcome': ['Pass' if i==0 else 'Fail' for i in y]}); fig_scatter = px.scatter(df_plot, x='X1', y='X2', color='Outcome', color_discrete_map={'Pass':'green', 'Fail':'red'});
    for trace in fig_scatter.data: fig.add_trace(trace)
    fig.update_layout(title_text='Predictive QC: Predicting Run Failure', xaxis_title='Reagent Age (days)', yaxis_title='Incubation Temp (°C)', height=600); return fig

def plot_forecasting():
    np.random.seed(42); dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W')); trend = np.linspace(0, 5, 104); seasonality = 1.5 * np.sin(np.arange(104) * (2 * np.pi / 52.14)); noise = np.random.normal(0, 0.5, 104); y = 50 + trend + seasonality + noise; df = pd.DataFrame({'ds': dates, 'y': y}); model = Prophet(weekly_seasonality=False, daily_seasonality=False); model.fit(df); future = model.make_future_dataframe(periods=26, freq='W'); forecast = model.predict(future)
    fig1 = plot_plotly(model, forecast); fig1.add_hline(y=58, line_dash="dash", line_color="red", annotation_text="Upper Spec Limit"); fig1.update_layout(title_text='Time Series Forecasting of Control Performance'); fig2 = plot_components_plotly(model, forecast); return fig1, fig2

def plot_wilson(successes, n_samples):
    p_hat = successes / n_samples if n_samples > 0 else 0; wald_lower, wald_upper = stats.norm.interval(0.95, loc=p_hat, scale=np.sqrt(p_hat*(1-p_hat)/n_samples)) if n_samples > 0 else (0,0); wilson_lower, wilson_upper = wilson_score_interval(p_hat, n_samples); cp_lower, cp_upper = stats.beta.interval(0.95, successes, n_samples - successes + 1) if n_samples > 0 else (0,1); intervals = {"Wald (Approximate)": (wald_lower, wald_upper, 'red'), "Wilson Score": (wilson_lower, wilson_upper, 'blue'), "Clopper-Pearson (Exact)": (cp_lower, cp_upper, 'green')}; fig = go.Figure()
    for i, (name, (lower, upper, color)) in enumerate(intervals.items()): fig.add_trace(go.Scatter(x=[lower, upper], y=[name, name], mode='lines+markers', line=dict(color=color, width=10), name=name, hoverinfo='text', text=f"[{lower:.3f}, {upper:.3f}]"))
    fig.add_vline(x=p_hat, line_dash="dash", line_color="black", annotation_text=f"Observed Rate={p_hat:.2%}"); fig.update_layout(title_text=f'Comparing 95% CIs for {successes}/{n_samples} Concordant Results', xaxis_title='Concordance Rate', height=500); return fig

# Replace the old plot_bayesian function with this one.
def plot_bayesian(prior_type):
    n_qc, successes_qc = 20, 18; observed_rate = successes_qc / n_qc;
    if prior_type == "Strong R&D Prior": prior_alpha, prior_beta = 490, 10
    elif prior_type == "Skeptical/Regulatory Prior": prior_alpha, prior_beta = 10, 10
    else: prior_alpha, prior_beta = 1, 1
    
    p_range = np.linspace(0.6, 1.0, 500)
    # Calculate Prior
    prior_dist = beta.pdf(p_range, prior_alpha, prior_beta)
    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    
    # Calculate Likelihood
    likelihood = stats.binom.pmf(k=successes_qc, n=n_qc, p=p_range)
    
    # Calculate Posterior
    posterior_alpha, posterior_beta = prior_alpha + successes_qc, prior_beta + (n_qc - successes_qc)
    posterior_dist = beta.pdf(p_range, posterior_alpha, posterior_beta)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # Create Plot
    fig = go.Figure()
    
    # Normalize for visualization
    max_y = np.max(posterior_dist)
    
    # Plot Likelihood
    fig.add_trace(go.Scatter(x=p_range, y=likelihood * max_y / np.max(likelihood), mode='lines', name='Likelihood (from QC Data)', line=dict(dash='dot', color='red'), fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'))
    
    # Plot Prior
    fig.add_trace(go.Scatter(x=p_range, y=prior_dist, mode='lines', name='Prior Belief', line=dict(dash='dash', color='green')))

    # Plot Posterior
    fig.add_trace(go.Scatter(x=p_range, y=posterior_dist, mode='lines', name='Posterior Belief', line=dict(color='blue', width=4), fill='tozeroy', fillcolor='rgba(0,0,255,0.2)'))

    fig.add_vline(x=posterior_mean, line_dash="solid", line_color="blue", annotation_text=f"Posterior Mean={posterior_mean:.3f}")
    
    fig.update_layout(
        title_text='Bayesian Inference: How Evidence Updates Belief',
        xaxis_title='Assay Pass Rate (Concordance)', yaxis_title='Probability Density / Scaled Likelihood',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig, prior_mean, observed_rate, posterior_mean
def plot_ci_concept():
    np.random.seed(123); pop_mean, pop_std, n = 100, 15, 30; n_sims = 100; capture_count = 0; fig = go.Figure()
    for i in range(n_sims):
        sample = np.random.normal(pop_mean, pop_std, n); sample_mean = np.mean(sample); margin_of_error = 1.96 * (pop_std / np.sqrt(n)); ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error; color = 'cornflowerblue' if ci_lower <= pop_mean <= ci_upper else 'red';
        if color == 'cornflowerblue': capture_count += 1
        fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[i, i], mode='lines', line=dict(color=color, width=3), hoverinfo='none')); fig.add_trace(go.Scatter(x=[sample_mean], y=[i], mode='markers', marker=dict(color='black', size=5), hoverinfo='none'))
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}"); fig.update_layout(title_text=f'Conceptual Simulation of 100 95% Confidence Intervals', xaxis_title='Value', yaxis_title='Simulation Run', showlegend=False, height=700); return fig, capture_count, n_sims


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

elif "Gage R&R" in method_key:
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
    # ... (Content for this method)
    st.markdown("**Purpose:** To verify the assay's ability to provide results that are directly proportional to the analyte concentration across a specified range. **Application:** This study establishes the validated 'reportable range' of the assay.")
    col1, col2 = st.columns([0.65, 0.35])
    with col1: fig, model = plot_linearity(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: R-squared (R²)", value=f"{model.rsquared:.4f}"); st.metric(label="💡 Metric: Slope", value=f"{model.params[1]:.3f}"); st.metric(label="💡 Metric: Y-Intercept", value=f"{model.params[0]:.2f}")
            st.markdown("- **Residual Plot:** The random scatter of points confirms the linear model is appropriate.")
            st.markdown("**The Bottom Line:** A high R², a slope near 1, and an intercept near 0 provide statistical proof that your assay behaves like a well-calibrated instrument across its intended range.")
        with tab2:
            st.markdown("- **R² > 0.995** is typically required."); st.markdown("- **Slope** should be close to 1.0 (e.g., within **0.95 - 1.05**)."); st.markdown("- The **95% CI for the Intercept** should contain **0**.")
        with tab3:
            st.markdown("**Origin:** Based on Ordinary Least Squares (OLS) regression (Legendre & Gauss, early 1800s)."); st.markdown("**Mathematical Basis:** We fit the model and test the hypotheses $H_0: \\beta_1 = 1$ and $H_0: \\beta_0 = 0$."); st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")

elif "LOD & LOQ" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To determine the lowest concentration at which the assay can reliably detect (LOD) and accurately quantify (LOQ) an analyte. **Application:** This defines the lower limit of the assay's useful range, critical for impurity testing or low-level biomarker detection.")
    col1, col2 = st.columns([0.65, 0.35]);
    with col1: fig, lod_val, loq_val = plot_lod_loq(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} units"); st.metric(label="💡 Metric: Limit of Detection (LOD)", value=f"{lod_val:.2f} units")
            st.markdown("- **LOD:** Answers 'Is the analyte present?'"); st.markdown("- **LOQ:** The lowest point in the reportable range.")
            st.markdown("**The Bottom Line:** This analysis defines the absolute floor of your assay's capability. It proves you can trust measurements down to the LOQ, and detect presence down to the LOD.")
        with tab2:
            st.markdown("- The **LOQ must be ≤ the lowest required concentration** for the assay's intended use.")
        with tab3:
            st.markdown("**Origin:** Based on International Council for Harmonisation (ICH) Q2(R1) guidelines."); st.markdown("**Mathematical Basis:** Uses the standard deviation of blank samples ($\sigma_{blank}$)."); st.latex("LOD = \\bar{y}_{blank} + 3.3 \\sigma_{blank}"); st.latex("LOQ = \\bar{y}_{blank} + 10 \\sigma_{blank}")

elif "Method Comparison" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To formally assess the agreement and bias between two methods (e.g., R&D vs. QC lab). **Application:** This is a cornerstone of transfer, replacing simpler tests with a more powerful analysis across the full measurement range.")
    col1, col2 = st.columns([0.65, 0.35])
    with col1: fig, slope, intercept, bias, ua, la = plot_method_comparison(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Mean Bias (B-A)", value=f"{bias:.2f}"); st.metric(label="💡 Metric: Deming Slope", value=f"{slope:.3f}"); st.metric(label="💡 Metric: Deming Intercept", value=f"{intercept:.2f}")
            st.markdown("- **Deming:** Checks for systematic constant (intercept) and proportional (slope) errors."); st.markdown("- **Bland-Altman:** Visualizes the random error and quantifies the expected range of disagreement.")
            st.markdown("**The Bottom Line:** Passing both analyses proves that the receiving lab's method is statistically indistinguishable from the reference method, confirming a successful transfer.")
        with tab2:
            st.markdown("- **Deming:** Slope CI should contain 1; Intercept CI should contain 0."); st.markdown(f"- **Bland-Altman:** >95% of points must be within the Limits of Agreement. The LoA width (`{la:.2f}` to `{ua:.2f}`) must be practically acceptable.")
        with tab3:
            st.markdown("**Origin:** Deming Regression (W. Edwards Deming); Bland-Altman plot (1986)."); st.markdown("**Mathematical Basis:** Deming minimizes perpendicular distances to the line. Bland-Altman plots Difference vs. Average; Limits of Agreement are $\\bar{d} \\pm 1.96 \\cdot s_d$.")

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
    # ... (Content for this method)
    st.markdown("**Purpose:** To implement sensitive charts that can detect small, systematic drifts or shifts in assay performance that a Shewhart chart might miss. **Application:** Long-term monitoring of controls to detect gradual reagent degradation or instrument drift.")
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM')); col1, col2 = st.columns([0.65, 0.35])
    with col1:
        if chart_type == 'EWMA': lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05); st.plotly_chart(plot_ewma_cusum(chart_type, lmbda, 0, 0), use_container_width=True)
        else: k_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25); H_sigma = st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5); st.plotly_chart(plot_ewma_cusum(chart_type, 0, k_sigma, H_sigma), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Shift Detection", value="Signal Detected", delta="Action Required", delta_color="inverse"); st.markdown("- **EWMA:** Best for detecting small, gradual *drifts*."); st.markdown("- **CUSUM:** Best for detecting small, *abrupt and sustained* shifts.")
            st.markdown("**The Bottom Line:** These are your early-warning systems. They catch the Villain (Variation) when it's being subtle, before it causes a major out-of-spec event.")
        with tab2:
            st.markdown("- **EWMA Rule:** For long-term monitoring, use a small `λ` (e.g., **0.1 to 0.3**)."); st.markdown("- **CUSUM Rule:** Set `k` to half the magnitude of the shift to detect.")
        with tab3:
            st.markdown("**Origin:** EWMA (Roberts, 1959); CUSUM (Page, 1954)."); st.markdown("**Mathematical Basis:** EWMA: $z_i = \\lambda x_i + (1-\\lambda)z_{i-1}$. CUSUM: $SH_i = \\max(0, SH_{i-1} + (x_i - \\mu_0) - k)$.")

elif "Run Validation" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To create an objective, statistically-driven system for accepting or rejecting each analytical run based on QC sample performance. **Application:** Routine QC in a regulated environment.")
    st.plotly_chart(plot_multi_rule(), use_container_width=True)
    st.subheader("Standard Industry Rule Sets")
    tab1, tab2, tab3 = st.tabs(["✅ Westgard Rules", "✅ Nelson Rules", "✅ Western Electric Rules"])
    with tab1: st.markdown("""Developed for lab QC, vital for CLIA, CAP, ISO 15189 compliance. A run is rejected if a "Rejection Rule" is violated.
| Rule | Use Case | Interpretation |
|---|---|---|
| **1_2s** | Warning | One control > ±2σ. Triggers inspection. |
| **1_3s** | Rejection | One control > ±3σ. |
| **2_2s** | Rejection | Two consecutive > same ±2σ limit. |
| **R_4s** | Rejection | One > +2σ and the next > -2σ. |
| **4_1s** | Rejection | Four consecutive > same ±1σ limit. |
| **10x** | Rejection | Ten consecutive points on the same side of the mean. |""")
    with tab2: st.markdown("""Excellent for catching non-random patterns in manufacturing and general SPC.
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
    with tab3: st.markdown("""Foundational rules from which many other systems were derived.
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

elif "Anomaly Detection" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To leverage machine learning to detect complex, multivariate anomalies that traditional univariate control charts would miss. **Application:** Proactive, real-time monitoring of complex assays to find novel failure modes.")
    col1, col2 = st.columns([0.65, 0.35])
    with col1: st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Anomalies Detected", value="3"); st.markdown("- This method finds 'unknown unknowns' by learning the normal multi-dimensional shape of the data."); st.markdown("- **The 'Holy Shit' Moment:** This is the 'ghost in the machine.' An operator swears every parameter is in spec, but the ML model flags a run as anomalous because the *combination* of parameters is wrong. This uncovers subtle failure modes no human could see.")
        with tab2:
            st.markdown("- Any point flagged as an **anomaly must be investigated** by SMEs to determine root cause.")
        with tab3:
            st.markdown("**Origin:** Proposed by Liu, Ting, and Zhou in 2008."); st.markdown("**Mathematical Basis:** Uses random trees to isolate points. The score $ s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}} $ is based on the average path length $E(h(x))$ to isolate a point.")

elif "Predictive QC" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To move from reactive to proactive quality control by predicting run failure based on in-process parameters *before* the run is completed. **Application:** A real-time decision support tool for lab operators.")
    col1, col2 = st.columns([0.65, 0.35])
    with col1: st.plotly_chart(plot_predictive_qc(), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Predictive Risk Profiling", value="Enabled"); st.markdown("- This model predicts the probability of a run failing based on its initial parameters."); st.markdown("- The color gradient shows the learned 'risk zones'.")
            st.markdown("**The Bottom Line:** This is the ultimate defense against the Villain of wasted resources. It stops bad runs before they even start.")
        with tab2:
            st.markdown("- A risk threshold is set, e.g., 'If **P(Fail) > 20%**, flag run for operator review.'")
        with tab3:
            st.markdown("**Origin:** Logistic regression was developed by David Cox in 1958."); st.markdown("**Mathematical Basis:** Models binary outcome probability using the sigmoid function: $ P(y=1|x) = 1 / (1 + e^{-(\\beta_0 + \\beta_1 x_1 + ...)}) $.")

elif "Control Forecasting" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To forecast the future performance of assay controls to anticipate problems. **Application:** Proactive scheduling of instrument maintenance or ordering of new reagent lots before performance degrades.")
    fig1_fc, fig2_fc = plot_forecasting()
    st.plotly_chart(fig1_fc, use_container_width=True)
    with st.expander("Interpretation, Rules & Theory"):
        st.markdown("""- **📈 KPI: Forecasted Trend.** The components plot below reveals if there is a consistent upward or downward trend over time.
- **💡 Key Insight:** The forecast shows the expected future path and uncertainty interval of the control. The components plot decomposes this into trend and seasonality for root cause analysis.
- **✅ Acceptance Rule:** A "proactive alert" can be triggered if the **lower bound of the 80% forecast interval is predicted to cross a specification limit** within the forecast horizon.
- **📖 Method Theory:** Prophet is an open-source library from Facebook based on a decomposable model: $ y(t) = g(t) + s(t) + h(t) + \\epsilon_t $, with terms for trend, seasonality, and holidays.""")
        st.plotly_chart(fig2_fc, use_container_width=True)

elif "Pass/Fail Analysis" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To accurately calculate a confidence interval for a proportion. **Application:** Essential for validating qualitative assays (e.g., presence/absence) where the result is a simple pass or fail.")
    n_samples_wilson = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30); successes_wilson = st.sidebar.slider("Concordant Results", 0, n_samples_wilson, int(n_samples_wilson * 0.95))
    col1, col2 = st.columns([0.65, 0.35])
    with col1: st.plotly_chart(plot_wilson(successes_wilson, n_samples_wilson), use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}")
            st.markdown("- **Wilson & Clopper-Pearson** intervals are robust for small samples."); st.markdown("- The **Wald interval** is unreliable and should be avoided.")
        with tab2:
            st.markdown("- **The lower bound of the 95% Wilson Score CI must be ≥ the target concordance rate** (e.g., 90%).")
        with tab3:
            st.markdown("**Origin:** Wilson Score (1927) and Clopper-Pearson (1934) improve upon the standard Wald interval."); st.markdown("**Mathematical Basis (Wilson):** $ \\frac{1}{1 + z^2/n} \\left( \\hat{p} + \\frac{z^2}{2n} \\pm z \\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n} + \\frac{z^2}{4n^2}} \\right) $")

elif "Bayesian Inference" in method_key:
    st.markdown("""
    **Purpose:** To formally combine existing knowledge (the 'Prior') with new experimental data (the 'Likelihood') to arrive at an updated, more robust conclusion (the 'Posterior').
    
    **Application:** This is the Hero's secret weapon for efficiency. Instead of starting from scratch, the Hero can leverage the vast knowledge from the R&D lab to design smaller, smarter validation studies at the QC site. It answers the question: "Given what we already knew, what does this new data tell us?"
    """)
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    
    col1, col2 = st.columns([0.65, 0.35])
    
    with col1:
        # This function is now enhanced for world-class rendering.
        fig, prior_mean, mle, posterior_mean = plot_bayesian(prior_type_bayes)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Posterior Mean Rate", value=f"{posterior_mean:.3f}")
            st.metric(label="💡 Prior Mean Rate", value=f"{prior_mean:.3f}")
            st.metric(label="💡 Data (MLE)", value=f"{mle:.3f}")
            st.markdown("- **Prior (Green):** Our initial belief. A 'Strong' prior is narrow and confident; a 'Skeptical' prior is broad and uncertain.")
            st.markdown("- **Likelihood (Red):** The evidence provided by the new data, sharply peaked at the observed rate.")
            st.markdown("- **Posterior (Blue):** The final, updated belief. It's a weighted compromise, pulled from the Prior towards the Likelihood.")
            st.markdown("**The Bottom Line:** The plot now tells a story. Notice how the strong prior isn't swayed much by the new data, while the skeptical prior is almost entirely convinced by it. This is Bayesian updating in action.")
        with tab2:
            st.markdown("- The **95% credible interval must be entirely above the target** (e.g., 90%).")
            st.markdown("- This approach allows for demonstrating success with smaller sample sizes if a strong, justifiable prior is used.")
        with tab3:
            st.markdown("**Origin:** Based on Bayes' Theorem (18th century), but made practical by modern computational methods.")
            st.markdown("**Mathematical Basis:** The core idea is that the posterior is proportional to the product of the likelihood and the prior.")
            st.latex("\\text{Posterior} \\propto \\text{Likelihood} \\times \\text{Prior}")
            st.markdown("For binomial data, we use the Beta-Binomial conjugate model: if the Prior is Beta($\\alpha, \\beta$) and we observe $k$ successes in $n$ trials, the Posterior is Beta($\\alpha + k, \\beta + n - k$).")
            
elif "Confidence Interval Concept" in method_key:
    # ... (Content for this method)
    st.markdown("**Purpose:** To understand the fundamental concept and correct interpretation of frequentist confidence intervals. **Application:** This is a foundational concept that underpins many of the statistical tests used in validation and quality control.")
    col1, col2 = st.columns([0.65, 0.35])
    with col1: fig, capture_count, n_sims = plot_ci_concept(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 The Golden Rule", "✅ Application", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Empirical Coverage", value=f"{(capture_count/n_sims):.0%}")
            st.markdown("A **95% confidence interval** means that if we were to repeat our experiment many times, **95% of the calculated intervals would contain the true, unknown parameter**. The confidence is in the *procedure*, not in any single interval.")
        with tab2:
            st.markdown("- This is a teaching module, not a validation step. The 'acceptance rule' is to **correctly interpret the CI** in reports and discussions.")
        with tab3:
            st.markdown("**Origin:** Introduced by Jerzy Neyman in the 1930s."); st.markdown("**Mathematical Basis:** $ \\text{CI} = \\text{Point Estimate} \\pm (\\text{Critical Value}) \\times (\\text{Standard Error}) $. For the mean with unknown $\\sigma$: $ \\bar{x} \\pm t_{\\alpha/2, n-1} \\frac{s}{\\sqrt{n}} $.")
