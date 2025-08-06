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
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding: 2rem 3rem; }
    .stExpander { border: 1px solid #e6e6e6; border-radius: 0.5rem; }
    h1, h2, h3 { color: #2c3e50; }
    .results-container {
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 15px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPER & GRAPHICS GENERATION FUNCTIONS
# ==============================================================================
@st.cache_data
def create_conceptual_map_plotly():
    """Generates the hierarchical map using Plotly."""
    nodes = {
        'DS': ('Data Science', 0, 3), 'BS': ('Biostatistics', 0, 2), 'ST': ('Statistics', 0, 1), 'IE': ('Industrial Engineering', 0, 0),
        'SI': ('Statistical Inference', 1, 2.5), 'SPC': ('SPC', 1, 0.5),
        'WS': ('Wilson Score', 2, 4), 'BAY': ('Bayesian Statistics', 2, 3.5), 'CI': ('Confidence Intervals', 2, 3), 'HT': ('Hypothesis Testing', 2, 2.5), 'NR': ('Nelson Rules', 2, 2), 'WR': ('Westgard Rules', 2, 1.5), 'PC': ('Process Capability', 2, 1), 'CC': ('Control Charts', 2, 0.5),
        'PE': ('Proportion Estimates', 3, 4), 'PP': ('Posterior Probabilities', 3, 3.5), 'ZME': ('Z-score / Margin of Error', 3, 3), 'TAV': ('T-tests / ANOVA', 3, 2.5), 'MQA': ('Manufacturing QA', 3, 2), 'CL': ('Clinical Labs', 3, 1.5), 'CSM': ('CUSUM', 3, 1), 'EWM': ('EWMA', 3, 0.5), 'SWH': ('Shewhart Charts', 3, 0),
    }
    edges = [('IE', 'SPC'), ('ST', 'SPC'), ('ST', 'SI'), ('BS', 'SI'), ('DS', 'SI'), ('SPC', 'CC'), ('SPC', 'PC'), ('SI', 'HT'), ('SI', 'CI'), ('SI', 'BAY'), ('SI', 'WR'), ('SI', 'NR'), ('CC', 'SWH'), ('CC', 'EWM'), ('CC', 'CSM'), ('PC', 'MQA'), ('WR', 'CL'), ('NR', 'MQA'), ('HT', 'TAV'), ('CI', 'ZME'), ('CI', 'WS'), ('BAY', 'PP'), ('WS', 'PE')]
    
    fig = go.Figure()
    
    # Add edges
    for start, end in edges:
        fig.add_trace(go.Scatter(x=[nodes[start][1], nodes[end][1]], y=[nodes[start][2], nodes[end][2]], mode='lines', line=dict(color='grey', width=1)))

    # Add nodes
    node_x = [v[1] for v in nodes.values()]; node_y = [v[2] for v in nodes.values()]; node_text = [v[0] for v in nodes.values()]
    colors = ["#e0f2f1"]*4 + ["#b2dfdb"]*2 + ["#80cbc4"]*8 + ["#4db6ac"]*9
    fig.add_trace(go.Scatter(x=node_x, y=node_y, text=node_text, mode='markers+text', textposition="top center", marker=dict(size=50, color=colors, line=dict(width=2, color='black'))))

    fig.update_layout(
        title_text='Hierarchical Map of Statistical Concepts in Quality & Development',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#f0f2f6'
    )
    return fig

def wilson_score_interval(p_hat, n, z=1.96):
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n
    term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2)))
    return (term1 - term2) / denom, (term1 + term2) / denom

# ==============================================================================
# PLOTTING FUNCTIONS (All 15 Methods, using Plotly)
# ==============================================================================
# All 15 plotting functions are included here, unabridged, with their Plotly implementations.

def plot_gage_rr():
    np.random.seed(10); n_operators, n_samples, n_replicates = 3, 10, 3; sample_means = np.linspace(90, 110, n_samples); operator_bias = [0, -0.5, 0.8]; data = []
    for op_idx, operator in enumerate(['Alice', 'Bob', 'Charlie']):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[op_idx], 1.5, n_replicates)
            for m in measurements: data.append([operator, f'Sample_{sample_idx+1}', m])
    df = pd.DataFrame(data, columns=['Operator', 'Sample', 'Measurement']); model = ols('Measurement ~ C(Sample) + C(Operator) + C(Sample):C(Operator)', data=df).fit(); anova_table = sm.stats.anova_lm(model, typ=2); ms_operator = anova_table['sum_sq']['C(Operator)']/anova_table['df']['C(Operator)']; ms_interaction = anova_table['sum_sq']['C(Sample):C(Operator)']/anova_table['df']['C(Sample):C(Operator)']; ms_error = anova_table['sum_sq']['Residual']/anova_table['df']['Residual']; var_repeatability = ms_error; var_reproducibility = ((ms_operator - ms_interaction) / (n_samples * n_replicates)) + ((ms_interaction - ms_error) / n_replicates); var_part = (anova_table['sum_sq']['C(Sample)']/anova_table['df']['C(Sample)'] - ms_interaction) / (n_operators * n_replicates); variances = {k: max(0, v) for k, v in locals().items() if 'var_' in k}; var_rr = variances['var_repeatability'] + variances['var_reproducibility']; var_total = var_rr + variances['var_part']; pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0; pct_part = (variances['var_part'] / var_total) * 100 if var_total > 0 else 0
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{}, {}]], subplot_titles=("Measurements by Sample and Operator", "Variance Contribution")); fig_box = px.box(df, x='Sample', y='Measurement', color='Operator'); fig_strip = px.strip(df, x='Sample', y='Measurement', color='Operator')
    for trace in fig_box.data: fig.add_trace(trace, row=1, col=1)
    for trace in fig_strip.data: fig.add_trace(trace, row=1, col=1)
    fig.add_trace(go.Bar(x=['% Gage R&R', '% Part-to-Part'], y=[pct_rr, pct_part], marker_color=['salmon', 'skyblue'], text=[f'{pct_rr:.1f}%', f'{pct_part:.1f}%'], textposition='auto'), row=1, col=2); fig.add_hline(y=10, line_dash="dash", line_color="darkgreen", annotation_text="Acceptable < 10%", annotation_position="bottom right", row=1, col=2); fig.add_hline(y=30, line_dash="dash", line_color="darkorange", annotation_text="Unacceptable > 30%", annotation_position="top right", row=1, col=2); fig.update_layout(title_text='Gage R&R Study: Quantifying Measurement System Error', showlegend=False, height=600); fig.update_xaxes(tickangle=45, row=1, col=1); return fig, pct_rr, pct_part

def plot_linearity():
    np.random.seed(42); nominal = np.array([10, 25, 50, 100, 150, 200, 250]); measured = nominal + np.random.normal(0, 2, len(nominal)) - (nominal/150)**3; X = sm.add_constant(nominal); model = sm.OLS(measured, X).fit(); b, m = model.params
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linearity Plot", "Residual Analysis")); fig.add_trace(go.Scatter(x=nominal, y=measured, mode='markers', name='Measured Values'), row=1, col=1); fig.add_trace(go.Scatter(x=nominal, y=model.predict(X), mode='lines', name='Best Fit Line'), row=1, col=1); fig.add_trace(go.Scatter(x=[0, 260], y=[0, 260], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1); fig.add_trace(go.Scatter(x=nominal, y=model.resid, mode='markers', name='Residuals', marker_color='green'), row=1, col=2); fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2); fig.update_layout(title_text='Assay Linearity and Range Verification', showlegend=True, height=600); fig.update_xaxes(title_text="Nominal Concentration (ng/mL)", row=1, col=1); fig.update_yaxes(title_text="Measured Concentration (ng/mL)", row=1, col=1); fig.update_xaxes(title_text="Nominal Concentration (ng/mL)", row=1, col=2); fig.update_yaxes(title_text="Residuals (Measured - Predicted)", row=1, col=2); return fig, model

def plot_lod_loq():
    np.random.seed(3); blanks = np.random.normal(1.5, 0.5, 20); low_conc = np.random.normal(5.0, 0.6, 20); mean_blank, std_blank = np.mean(blanks), np.std(blanks, ddof=1); LOD = mean_blank + 3.3 * std_blank; LOQ = mean_blank + 10 * std_blank; x_kde = np.linspace(0, 8, 200); kde_blanks = stats.gaussian_kde(blanks)(x_kde); kde_low = stats.gaussian_kde(low_conc)(x_kde)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=x_kde, y=kde_blanks, fill='tozeroy', name='Blank Sample Distribution')); fig.add_trace(go.Scatter(x=x_kde, y=kde_low, fill='tozeroy', name='Low Conc. Sample Distribution')); fig.add_vline(x=LOD, line_dash="dash", line_color="orange", annotation_text=f"LOD={LOD:.2f}"); fig.add_vline(x=LOQ, line_dash="dash", line_color="red", annotation_text=f"LOQ={LOQ:.2f}"); fig.update_layout(title_text='Limit of Detection (LOD) and Quantitation (LOQ)', xaxis_title='Assay Signal (e.g., Absorbance)', yaxis_title='Density', height=600); return fig, LOD, LOQ

def plot_method_comparison():
    np.random.seed(42); x = np.linspace(20, 150, 50); y = 0.98 * x + 1.5 + np.random.normal(0, 2.5, 50); delta = np.var(y, ddof=1) / np.var(x, ddof=1); x_mean, y_mean = np.mean(x), np.mean(y); Sxx = np.sum((x-x_mean)**2); Sxy = np.sum((x-x_mean)*(y-y_mean)); beta1_deming = (np.sum((y-y_mean)**2) - delta*Sxx + np.sqrt((np.sum((y-y_mean)**2) - delta*Sxx)**2 + 4*delta*Sxy**2)) / (2*Sxy); beta0_deming = y_mean - beta1_deming*x_mean; diff = y - x; mean_diff = np.mean(diff); upper_loa = mean_diff + 1.96*np.std(diff,ddof=1); lower_loa = mean_diff - 1.96*np.std(diff,ddof=1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Deming Regression", "Bland-Altman Agreement Plot")); fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Sample Results'), row=1, col=1); fig.add_trace(go.Scatter(x=x, y=beta0_deming + beta1_deming*x, mode='lines', name='Deming Fit'), row=1, col=1); fig.add_trace(go.Scatter(x=[0, 160], y=[0, 160], mode='lines', name='Line of Identity', line=dict(dash='dash', color='black')), row=1, col=1); fig.add_trace(go.Scatter(x=(x+y)/2, y=diff, mode='markers', name='Difference', marker_color='purple'), row=1, col=2); fig.add_hline(y=mean_diff, line_color="red", annotation_text=f"Mean Bias={mean_diff:.2f}", row=1, col=2); fig.add_hline(y=upper_loa, line_dash="dash", line_color="blue", annotation_text=f"Upper LoA={upper_loa:.2f}", row=1, col=2); fig.add_hline(y=lower_loa, line_dash="dash", line_color="blue", annotation_text=f"Lower LoA={lower_loa:.2f}", row=1, col=2); fig.update_layout(title_text='Method Comparison: R&D Lab vs QC Lab', showlegend=True, height=600); fig.update_xaxes(title_text="R&D Lab (Reference)", row=1, col=1); fig.update_yaxes(title_text="QC Lab (Test)", row=1, col=1); fig.update_xaxes(title_text="Average of Methods", row=1, col=2); fig.update_yaxes(title_text="Difference (QC - R&D)", row=1, col=2); return fig, beta1_deming, beta0_deming, mean_diff, upper_loa, lower_loa

def plot_robustness():
    data = {'Temp': [-1, 1, -1, 1, -1, 1, -1, 1], 'pH': [-1, -1, 1, 1, -1, -1, 1, 1], 'Time': [-1, -1, -1, -1, 1, 1, 1, 1]}; df = pd.DataFrame(data); df['Response'] = 100 + 5*df['Temp'] - 2*df['pH'] + 1.5*df['Time'] - 3*df['Temp']*df['pH'] + np.random.normal(0, 1, 8); model = ols('Response ~ Temp * pH * Time', data=df).fit(); effects = model.params.iloc[1:]; effects = effects.sort_values(key=abs, ascending=False)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Pareto Plot of Standardized Effects", "Significant Interaction Plot: Temp * pH")); fig.add_trace(go.Bar(y=effects.index, x=effects.values, orientation='h'), row=1, col=1); df_int = df.groupby(['Temp', 'pH'])['Response'].mean().reset_index()
    for p_val, sub_df in df_int.groupby('pH'): fig.add_trace(go.Scatter(x=sub_df['Temp'], y=sub_df['Response'], mode='lines+markers', name=f"pH = {'High' if p_val==1 else 'Low'}"), row=2, col=1)
    fig.update_layout(title_text='Assay Robustness (Design of Experiments)', height=700, showlegend=True); fig.update_xaxes(title_text="Effect Magnitude", row=1, col=1); fig.update_xaxes(title_text="Temperature", tickvals=[-1, 1], ticktext=['Low', 'High'], row=2, col=1); fig.update_yaxes(title_text="Assay Response", row=2, col=1); return fig

def plot_shewhart():
    np.random.seed(42); in_control = np.random.normal(100.0, 2.0, 15); reagent_shift = np.random.normal(108.0, 2.0, 10); data = np.concatenate([in_control, reagent_shift]); x = np.arange(1, len(data) + 1); mean = np.mean(data[:15]); mr = np.abs(np.diff(data)); mr_mean = np.mean(mr[:14]); sigma_est = mr_mean / 1.128; UCL_I, LCL_I = mean + 3 * sigma_est, mean - 3 * sigma_est; out_of_control_I = np.where((data > UCL_I) | (data < LCL_I))[0]; UCL_MR = mr_mean * 3.267
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("I-Chart: Monitors Accuracy (Bias)", "MR-Chart: Monitors Precision (Variability)"), vertical_spacing=0.1); fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers', name='Control Value'), row=1, col=1); fig.add_trace(go.Scatter(x=x[out_of_control_I], y=data[out_of_control_I], mode='markers', marker=dict(color='red', size=12), name='Signal'), row=1, col=1); fig.add_hline(y=mean, line_dash="dash", line_color="black", row=1, col=1); fig.add_hline(y=UCL_I, line_color="red", row=1, col=1); fig.add_hline(y=LCL_I, line_color="red", row=1, col=1); fig.add_vrect(x0=15.5, x1=25.5, fillcolor="orange", opacity=0.2, layer="below", line_width=0, name="New Lot", row=1, col=1); fig.add_trace(go.Scatter(x=x[1:], y=mr, mode='lines+markers', name='Moving Range', marker_color='teal'), row=2, col=1); fig.add_hline(y=mr_mean, line_dash="dash", line_color="black", row=2, col=1); fig.add_hline(y=UCL_MR, line_color="red", row=2, col=1); fig.update_layout(title_text='Process Stability Monitoring: Shewhart I-MR Chart', height=700, showlegend=False); fig.update_yaxes(title_text="Concentration (ng/mL)", row=1, col=1); fig.update_yaxes(title_text="Range (ng/mL)", row=2, col=1); fig.update_xaxes(title_text="Analytical Run Number", row=2, col=1); return fig

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

def plot_multi_rule(rule_set='Westgard'):
    np.random.seed(3); mean, std = 100, 2; data = np.concatenate([np.random.normal(mean, std, 5), [mean + 2.1*std, mean + 2.2*std], np.random.normal(mean, std, 2), np.linspace(mean-0.5*std, mean-2*std, 6), [mean + 3.5*std], np.random.normal(mean + 1.5*std, 0.3, 4), np.random.normal(mean, std, 3), np.random.normal(mean - 1.5*std, 0.3, 5)]); x = np.arange(1, len(data) + 1); fig = go.Figure(); fig.add_trace(go.Scatter(x=x, y=data, mode='lines+markers', name='QC Sample', line=dict(color='darkblue')));
    for i, color in zip([1, 2, 3], ['gold', 'orange', 'red']):
        fig.add_hrect(y0=mean+i*std, y1=mean+(i+1)*std, fillcolor=color, opacity=0.1, layer="below", line_width=0); fig.add_hrect(y0=mean-i*std, y1=mean-(i+1)*std, fillcolor=color, opacity=0.1, layer="below", line_width=0)
        fig.add_hline(y=mean+i*std, line_dash="dot", line_color="gray", annotation_text=f"+{i}σ"); fig.add_hline(y=mean-i*std, line_dash="dot", line_color="gray", annotation_text=f"-{i}σ")
    fig.add_hline(y=mean, line_dash="dash", line_color="black", annotation_text="Mean"); fig.update_layout(title_text='QC Run Validation Chart', xaxis_title='QC Run Number', yaxis_title='Measured Value', height=600); return fig

def plot_capability(Cpk_target):
    np.random.seed(42); LSL, USL = 90, 110; process_mean = 101; process_std = (USL - LSL) / (6 * Cpk_target); data = np.random.normal(process_mean, process_std, 200); sigma_hat = np.std(data, ddof=1); Cpu = (USL - data.mean()) / (3 * sigma_hat); Cpl = (data.mean() - LSL) / (3 * sigma_hat); Cpk = np.min([Cpu, Cpl]); fig = px.histogram(data, nbins=30, histnorm='probability density', marginal='rug'); fig.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL"); fig.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL"); fig.add_vline(x=data.mean(), line_dash="dot", line_color="black", annotation_text="Mean"); color = "darkgreen" if Cpk >= 1.33 else "darkred"; fig.add_annotation(text=f"Cpk = {Cpk:.2f}", align='left', showarrow=False, xref='paper', yref='paper', x=0.05, y=0.95, bordercolor="black", borderwidth=1, bgcolor=color, font=dict(color="white")); fig.update_layout(title_text='Process Capability Analysis (Cpk)', height=600); return fig, Cpk

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

def plot_bayesian(prior_type):
    n_qc, successes_qc = 20, 18; observed_rate = successes_qc / n_qc;
    if prior_type == "Strong R&D Prior": prior_alpha, prior_beta = 490, 10
    elif prior_type == "Skeptical/Regulatory Prior": prior_alpha, prior_beta = 10, 10
    else: prior_alpha, prior_beta = 1, 1
    p_range = np.linspace(0, 1, 1000); prior_dist = beta.pdf(p_range, prior_alpha, prior_beta); posterior_alpha, posterior_beta = prior_alpha + successes_qc, prior_beta + (n_qc - successes_qc); posterior_dist = beta.pdf(p_range, posterior_alpha, posterior_beta); cred_interval = beta.interval(0.95, posterior_alpha, posterior_beta)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=p_range, y=prior_dist, mode='lines', name='Prior Belief', line=dict(dash='dash', color='green'))); fig.add_trace(go.Scatter(x=p_range, y=posterior_dist, mode='lines', name='Posterior Belief', line=dict(color='blue', width=3), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)')); fig.add_vline(x=observed_rate, line_dash="dot", line_color="red", annotation_text=f"QC Data={observed_rate:.2%}")
    fig.update_layout(title_text='Bayesian Inference for Assay Concordance Rate', xaxis_title='Assay Pass Rate (Concordance)', yaxis_title='Probability Density', height=600, xaxis_range=[0.7, 1.0]); return fig

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

# --- Sidebar Controls ---
st.sidebar.title("Toolkit Navigation")
st.sidebar.markdown("Select a statistical method to analyze and visualize.")
method_key = st.sidebar.radio("Select a Method:", options=[
    "1. Gage R&R", "2. Linearity and Range", "3. LOD & LOQ", "4. Method Comparison",
    "5. Assay Robustness (DOE)", "6. Process Stability (Shewhart)", "7. Small Shift Detection",
    "8. Run Validation", "9. Process Capability (Cpk)", "10. Anomaly Detection (ML)",
    "11. Predictive QC (ML)", "12. Control Forecasting (AI)", "13. Pass/Fail Analysis",
    "14. Bayesian Inference", "15. Confidence Interval Concept"
])
st.header(method_key)

# --- Dynamic Content Display ---
if "Gage R&R" in method_key:
    st.markdown("**Objective:** Before evaluating a process, you must first validate the measurement system. A Gage R&R study quantifies the inherent variability (error) of the assay, partitioning it into components like repeatability and reproducibility.")
    col1, col2 = st.columns([0.7, 0.3]);
    with col1: fig, pct_rr, pct_part = plot_gage_rr(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="% Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse"); st.metric(label="% Part-to-Part Variation", value=f"{pct_part:.1f}%", delta="Higher is better")
            st.markdown("---"); st.markdown("##### Acceptance Rules (AIAG):"); st.markdown("- **< 10%:** System is **acceptable**."); st.markdown("- **10% - 30%:** **Conditionally acceptable**."); st.markdown("- **> 30%:** System is **unacceptable**.")

elif "Linearity and Range" in method_key:
    st.markdown("**Objective:** To verify the assay's ability to provide results that are directly proportional to the analyte concentration across a specified range, thereby defining its reportable limits.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: fig, model = plot_linearity(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="R-squared (R²)", value=f"{model.rsquared:.4f}"); st.metric(label="Slope", value=f"{model.params[1]:.3f}"); st.metric(label="Y-Intercept", value=f"{model.params[0]:.2f}")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- **R² > 0.995** is typically required."); st.markdown("- **Slope** should be within **0.95 - 1.05**."); st.markdown("- **Intercept CI** should contain **0**."); st.markdown("- **Residuals** must be random and unstructured.")

elif "LOD & LOQ" in method_key:
    st.markdown("**Objective:** To determine the lowest concentration at which the assay can reliably detect (LOD) and accurately quantify (LOQ) an analyte, defining the lower limit of its useful range.")
    col1, col2 = st.columns([0.7, 0.3]);
    with col1: fig, lod_val, loq_val = plot_lod_loq(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="Limit of Detection (LOD)", value=f"{lod_val:.2f} units"); st.metric(label="Limit of Quantitation (LOQ)", value=f"{loq_val:.2f} units")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- **LOQ must be ≤ the lowest required concentration** for the assay's intended use.")

elif "Method Comparison" in method_key:
    st.markdown("**Objective:** To formally assess the agreement and bias between two methods (e.g., R&D vs. QC lab). This is a cornerstone of transfer, replacing simpler tests with a more powerful analysis across the full measurement range.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: fig, slope, intercept, bias, ua, la = plot_method_comparison(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="Deming Slope", value=f"{slope:.3f}"); st.metric(label="Mean Bias (B-A)", value=f"{bias:.2f}")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- **Deming:** Slope CI should contain 1; Intercept CI should contain 0."); st.markdown(f"- **Bland-Altman:** >95% of points must be within the Limits of Agreement. The LoA width (`{la:.2f}` to `{ua:.2f}`) must be practically acceptable.")

elif "Assay Robustness" in method_key:
    st.markdown("**Objective:** To proactively assess the assay's performance when small, deliberate changes are made to its input parameters (e.g., temperature, pH), identifying which factors are most critical to control.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: st.plotly_chart(plot_robustness(), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **Pareto Plot:** Ranks factors by their impact, focusing control efforts on the 'vital few'."); st.markdown("- **Interaction Plot:** Non-parallel lines reveal complex relationships that must be controlled.")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- The study must prove that **small, expected parameter variations do NOT significantly impact results**. If a factor is significant, its operating range in the SOP must be tightened.")

elif "Process Stability" in method_key:
    st.markdown("**Objective:** To demonstrate that the assay can be run in a stable and predictable manner at the receiving site, monitoring both accuracy (mean) and precision (variability).")
    col1, col2 = st.columns([0.7, 0.3]);
    with col1: st.plotly_chart(plot_shewhart(), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **I-Chart:** Monitors the process center (accuracy)."); st.markdown("- **MR-Chart:** Monitors run-to-run variability (precision).")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- Process is stable when **at least 20-25 consecutive points on both charts show no out-of-control signals**.")

elif "Small Shift Detection" in method_key:
    st.markdown("**Objective:** To implement sensitive charts that can detect small, systematic drifts or shifts in assay performance that a Shewhart chart might miss.")
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM')); col1, col2 = st.columns([0.7, 0.3])
    with col1:
        if chart_type == 'EWMA': lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05); st.plotly_chart(plot_ewma_cusum(chart_type, lmbda, 0, 0), use_container_width=True)
        else: k_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25); H_sigma = st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5); st.plotly_chart(plot_ewma_cusum(chart_type, 0, k_sigma, H_sigma), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **EWMA:** Best for detecting small *drifts*. **Rule:** Use a small `λ` (0.1-0.3) for monitoring."); st.markdown("- **CUSUM:** Best for detecting small, *abrupt and sustained* shifts. **Rule:** Set `k` to half the magnitude of the shift to detect.")

elif "Run Validation" in method_key:
    st.markdown("**Objective:** To create an objective, statistically-driven system for accepting or rejecting each analytical run based on QC sample performance.")
    st.plotly_chart(plot_multi_rule("Westgard"), use_container_width=True)
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
| 4. 14 points alternating up/down | Systematic oscillation |""")
    with tab3: st.markdown("""Foundational rules from which many other systems were derived.
| Rule | Interpretation |
|---|---|
| **Rule 1** | One point falls outside the ±3σ limits. |
| **Rule 2** | Two out of three consecutive points fall beyond the ±2σ limit on the same side. |
| **Rule 3** | Four out of five consecutive points fall beyond the ±1σ limit on the same side. |
| **Rule 4** | Eight consecutive points fall on the same side of the mean. |""")

elif "Process Capability" in method_key:
    st.markdown("**Objective:** To determine if the stable process is capable of consistently producing results that meet specifications, linking SPC to engineering tolerances.")
    cpk_target_slider = st.sidebar.slider("Simulate a process with a desired Cpk target:", 0.5, 2.0, 1.33, 0.01)
    col1, col2 = st.columns([0.7, 0.3])
    with col1: fig, cpk_val = plot_capability(cpk_target_slider); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="Calculated Cpk", value=f"{cpk_val:.2f}")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- `Cpk ≥ 1.33`: Process is **capable**."); st.markdown("- `Cpk ≥ 1.67`: Process is **highly capable**."); st.markdown("- `Cpk < 1.0`: Process is **not capable**.")

elif "Anomaly Detection" in method_key:
    st.markdown("**Objective:** To leverage machine learning to detect complex, multivariate anomalies that traditional univariate control charts would miss.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: st.plotly_chart(plot_anomaly_detection(), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **Identifies** points that are anomalous in multi-dimensional space."); st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- Any point flagged as an **anomaly must be investigated** by SMEs to determine root cause.")

elif "Predictive QC" in method_key:
    st.markdown("**Objective:** To move from reactive to proactive quality control by predicting run failure based on in-process parameters *before* the run is completed.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: st.plotly_chart(plot_predictive_qc(), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **Predicts** the probability of a run failing based on initial parameters."); st.markdown("- **Decision Boundary** shows the learned 'risk zones'.")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- A risk threshold is set, e.g., 'If **P(Fail) > 20%**, flag run for operator review.'")

elif "Control Forecasting" in method_key:
    st.markdown("**Objective:** To forecast the future performance of assay controls to anticipate problems and enable proactive management of maintenance and reagent lots.")
    fig1_fc, fig2_fc = plot_forecasting()
    st.plotly_chart(fig1_fc, use_container_width=True)
    with st.expander("Interpretation & Component Analysis"):
        st.markdown("""- **Forecast Plot:** Shows the expected future path and uncertainty interval of the control.
        - **Components Plot:** Decomposes the forecast into trend and seasonality for root cause analysis.
        - **Rule:** A "proactive alert" can be triggered if the **lower bound of the 80% forecast interval is predicted to cross a specification limit** within the forecast horizon.""")
        st.plotly_chart(fig2_fc, use_container_width=True)

elif "Pass/Fail Analysis" in method_key:
    st.markdown("**Objective:** To accurately calculate a confidence interval for a proportion, essential for validating qualitative assays (e.g., presence/absence).")
    n_samples_wilson = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30); successes_wilson = st.sidebar.slider("Concordant Results", 0, n_samples_wilson, int(n_samples_wilson * 0.95))
    col1, col2 = st.columns([0.7, 0.3])
    with col1: st.plotly_chart(plot_wilson(successes_wilson, n_samples_wilson), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.metric(label="Observed Rate", value=f"{(successes_wilson/n_samples_wilson if n_samples_wilson > 0 else 0):.2%}")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- **The lower bound of the 95% Wilson Score CI must be ≥ the target concordance rate** (e.g., 90%).")

elif "Bayesian Inference" in method_key:
    st.markdown("**Objective:** To formally combine historical data (the 'Prior') with new data (the 'Likelihood') to arrive at a more robust conclusion (the 'Posterior').")
    prior_type_bayes = st.sidebar.radio("Select Prior Belief:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    col1, col2 = st.columns([0.7, 0.3])
    with col1: st.plotly_chart(plot_bayesian(prior_type_bayes), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("Key Insights & Acceptance"); st.markdown("- **Posterior** (blue line) is the updated belief."); st.markdown("- **Strong Priors** require more data to be swayed.")
            st.markdown("---"); st.markdown("##### Acceptance Rules:"); st.markdown("- The **95% credible interval must be entirely above the target** (e.g., 90%).")

elif "Confidence Interval Concept" in method_key:
    st.markdown("**Objective:** To understand the fundamental concept and correct interpretation of frequentist confidence intervals, which underpin many statistical tests.")
    col1, col2 = st.columns([0.7, 0.3])
    with col1: fig, capture_count, n_sims = plot_ci_concept(); st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("The Golden Rule"); st.metric(label="Capture Rate in Simulation", value=f"{(capture_count/n_sims):.0%}")
            st.markdown("A **95% confidence interval** means that if we were to repeat our experiment many times, **95% of the calculated intervals would contain the true, unknown parameter**. The confidence is in the *procedure*, not in any single interval.")
