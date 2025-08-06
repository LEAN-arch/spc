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
st.set_page_config(layout="wide", page_title="An Interactive Guide to Biotech Assay Technology Transfer Statistics & ML/AI", page_icon="📈")

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
def plot_v_model():
    """Generates a professional, interactive V&V Model diagram using Plotly."""
    fig = go.Figure()

    # Define the nodes, their positions, and detailed, universally applicable information
    v_model_stages = {
        'URS': {'name': 'User Requirements', 'x': 0, 'y': 5, 'question': 'What does the business/patient/process need?', 'tools': 'Business Case, User Needs Document'},
        'FS': {'name': 'Functional Specs', 'x': 1, 'y': 4, 'question': 'What must the system *do*?', 'tools': 'Assay: Linearity, LOD/LOQ. Instrument: Throughput. Software: User Roles.'},
        'DS': {'name': 'Design Specs', 'x': 2, 'y': 3, 'question': 'How will the system be built/configured?', 'tools': 'Assay: Robustness (DOE). Instrument: Component selection. Software: Architecture.'},
        'BUILD': {'name': 'Implementation', 'x': 3, 'y': 2, 'question': 'Build, code, configure, write SOPs, train.', 'tools': 'N/A (Physical/Code Transfer)'},
        'IQOQ': {'name': 'Installation/Operational Qualification (IQ/OQ)', 'x': 4, 'y': 3, 'question': 'Is the system installed correctly and does it operate as designed?', 'tools': 'Instrument Calibration, Software Unit/Integration Tests.'},
        'PQ': {'name': 'Performance Qualification (PQ)', 'x': 5, 'y': 4, 'question': 'Does the functioning system perform reliably in its environment?', 'tools': 'Gage R&R, Method Comp, Stability, Process Capability (Cpk).'},
        'UAT': {'name': 'User Acceptance / Validation', 'x': 6, 'y': 5, 'question': 'Does the validated system meet the original user need?', 'tools': 'Pass/Fail Analysis, Bayesian Confirmation, Final Report.'}
    }
    
    # Define colors
    verification_color = 'rgba(0, 128, 128, 0.8)'  # Teal
    validation_color = 'rgba(0, 191, 255, 0.8)'  # Deep Sky Blue
    
    # Create the 'V' shape with a bold line
    path_keys = ['URS', 'FS', 'DS', 'BUILD', 'IQOQ', 'PQ', 'UAT']
    path_x = [v_model_stages[p]['x'] for p in path_keys]
    path_y = [v_model_stages[p]['y'] for p in path_keys]
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y, mode='lines',
        line=dict(color='darkgrey', width=3),
        hoverinfo='none'
    ))

    # Add nodes as shapes for a cleaner look
    for i, (key, stage) in enumerate(v_model_stages.items()):
        color = verification_color if i < 3 else validation_color if i > 3 else 'grey'
        fig.add_shape(
            type="rect",
            x0=stage['x'] - 0.4, y0=stage['y'] - 0.25,
            x1=stage['x'] + 0.4, y1=stage['y'] + 0.25,
            line=dict(color="black", width=2),
            fillcolor=color,
        )
        fig.add_annotation(
            x=stage['x'], y=stage['y'],
            text=f"<b>{stage['name']}</b>",
            showarrow=False,
            font=dict(color='white', size=11)
        )
        # Add a transparent scatter marker on top for the hover tooltip
        fig.add_trace(go.Scatter(
            x=[stage['x']], y=[stage['y']],
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)', size=60),
            hoverinfo='text',
            text=f"<b>{stage['name']}</b><br><br><i>{stage['question']}</i><br><b>Examples / Tools:</b> {stage['tools']}"
        ))

    # Add horizontal verification/validation lines
    for i in range(3):
        start_key = path_keys[i]
        end_key = path_keys[-(i+1)]
        fig.add_shape(
            type="line",
            x0=v_model_stages[start_key]['x'], y0=v_model_stages[start_key]['y'],
            x1=v_model_stages[end_key]['x'], y1=v_model_stages[end_key]['y'],
            line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot")
        )

    fig.update_layout(
        title_text='<b>The V&V Model for Technology Transfer (Hover for Details)</b>',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1.5, 5.7]),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#f0f2f6'
    )
    
    return fig, v_model_stages

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

def plot_ci_concept(n=30):
    # --- Data Generation ---
    np.random.seed(123)
    pop_mean, pop_std = 100, 15
    n_sims = 100
    
    # Generate population for visualization
    population = np.random.normal(pop_mean, pop_std, 10000)
    
    # Generate many sample means for the sampling distribution
    sample_means = [np.mean(np.random.normal(pop_mean, pop_std, n)) for _ in range(1000)]
    
    # --- Figure Creation (Multi-plot Dashboard) ---
    
    # --- Plot 1: Distributions (using KDE for smooth curves) ---
    fig1 = go.Figure()
    
    # KDE for Population
    kde_pop = stats.gaussian_kde(population)
    x_range_pop = np.linspace(population.min(), population.max(), 500)
    fig1.add_trace(go.Scatter(
        x=x_range_pop, y=kde_pop(x_range_pop),
        fill='tozeroy', name='True Population Distribution',
        marker_color='skyblue', opacity=0.6,
        hoverinfo='none'
    ))
    
    # KDE for Sampling Distribution
    kde_means = stats.gaussian_kde(sample_means)
    x_range_means = np.linspace(min(sample_means), max(sample_means), 500)
    fig1.add_trace(go.Scatter(
        x=x_range_means, y=kde_means(x_range_means),
        fill='tozeroy', name=f'Distribution of Sample Means (n={n})',
        marker_color='darkorange', opacity=0.6,
        hoverinfo='none'
    ))
    
    # Add a marker for "Our One Sample"
    our_sample_mean = sample_means[0]
    fig1.add_trace(go.Scatter(
        x=[our_sample_mean], y=[0], mode='markers', name='Our One Sample Mean',
        marker=dict(color='black', size=12, symbol='x'),
        hovertemplate=f"Our Sample Mean: {our_sample_mean:.2f}<extra></extra>"
    ))
    
    fig1.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}")
    fig1.update_layout(
        title_text=f"<b>The Theoretical Universe (Sample Size n={n})</b>",
        yaxis_title="Density", xaxis_title="Value",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # --- Plot 2: Confidence Interval Simulation ---
    fig2 = go.Figure()
    capture_count = 0
    total_width = 0
    for i in range(n_sims):
        sample = np.random.normal(pop_mean, pop_std, n)
        sample_mean = np.mean(sample)
        margin_of_error = 1.96 * (pop_std / np.sqrt(n))
        ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error
        total_width += (ci_upper - ci_lower)
        
        color = 'cornflowerblue' if ci_lower <= pop_mean <= ci_upper else 'red'
        if color == 'cornflowerblue':
            capture_count += 1
        
        status = "Capture" if color == 'cornflowerblue' else "Miss"
        
        fig2.add_trace(go.Scatter(
            x=[ci_lower, ci_upper], y=[i, i], mode='lines',
            line=dict(color=color, width=3),
            hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Status: {status}<br>Interval: [{ci_lower:.2f}, {ci_upper:.2f}]<extra></extra>"
        ))
        
        fig2.add_trace(go.Scatter(
            x=[sample_mean], y=[i], mode='markers',
            marker=dict(color='black', size=5, symbol='line-ns-open'),
            hovertemplate=f"<b>Run {i+1} (n={n})</b><br>Sample Mean: {sample_mean:.2f}<extra></extra>"
        ))
    
    avg_width = total_width / n_sims
    fig2.add_vline(x=pop_mean, line_dash="dash", line_color="black", annotation_text=f"True Mean={pop_mean}")
    fig2.update_layout(
        title_text=f"<b>The Practical Result: 100 Simulated CIs (Sample Size n={n})</b>",
        yaxis_title="Simulation Run", xaxis_title="Value",
        showlegend=False,
        yaxis_range=[-2, n_sims+2]
    )
    
    return fig1, fig2, capture_count, n_sims, avg_width
# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
st.title("🛠️ The Guild: An Interactive Guide to V&V and Tech Transfer Using Tools from Statistics and Machine Learning.")
st.markdown("Welcome to this interactive guide. It's a collection of tools to help explore statistical and ML methods that help support a robust V&V, tech transfer, and lifecycle management plan, bridging classical SPC with modern ML/AI concepts.")

st.plotly_chart(create_conceptual_map_plotly(), use_container_width=True)
st.markdown("This map illustrates how foundational **Academic Disciplines** like Statistics, Machine Learning, and Industrial Engineering give rise to **Core Domains** such as Statistical Process Control (SPC) and Statistical Inference. These domains, in turn, provide the **Sub-Domains & Concepts** that are the basis for the **Specific Tools & Applications** you can explore in this guide. Use the sidebar to navigate through these practical applications.")
st.divider()

st.header("The Scientist's/Engineer's Journey: A Three-Act Story")
st.markdown("""In the world of quality and development, during v&v, and tech transfer, our story always has the same **Hero**: the dedicated scientist, engineer, or analyst. And it always has the same **Villain**: insidious, hidden, and costly **Variation**.
This toolkit is structured as a three-act journey to empower you to conquer the villain. Each method is a tool, a weapon, or a new sense to perceive and control the variation affecting your processes.""")
act1, act2, act3 = st.columns(3)
with act1: st.subheader("Act I: Know Thyself (The Foundation)"); st.markdown("Before the battle, the Hero must understand their own strengths and weaknesses. What is the true capability of their measurement system? What are its limits? This is the foundational work of **Characterization and Validation**.")
with act2: st.subheader("Act II: The Transfer (The Crucible)"); st.markdown("The Hero's validated method must now survive in a new land—the receiving QC lab. This is the ultimate test of **Robustness, Stability, and Comparability**. It is here that many battles with Variation are won or lost.")
with act3: st.subheader("Act III: The Guardian (Beyond the Transfer)"); st.markdown("The assay is live, but the Villain never sleeps. The Hero must now become a guardian, using advanced tools to **Monitor, Predict, and Protect** the process for its entire lifecycle, anticipating problems before they arise.")


# --- V&V Model and Narrative Introduction ---
st.subheader("The V&V Model: The Hero's Map")
st.markdown("""
The **Verification & Validation (V&V) Model**, shown below, is the universally accepted strategic framework that serves as the map for our Hero's journey. It applies to any technology transfer, whether it's an **assay, instrument, process, or software**.

- **The Left Side (Verification - "Are we building it right?"):** This is the journey *down* into the details, building a solid foundation.
- **The Right Side (Validation - "Did we build the right thing?"):** This is the journey *up* to prove that the system meets the original needs.

The tools in this toolkit are the specific steps you take to conquer each stage of this map. **Hover over a stage in the diagram to learn more.**
""")

fig_v_model, v_model_stages = plot_v_model()
st.plotly_chart(fig_v_model, use_container_width=True)

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

if "Gage R&R" in method_key:
    st.markdown("""
    **Purpose:** To quantify the inherent variability (error) of the measurement system itself, separating it from the actual process variation. 
    
    **Definition:** A Gage R&R study partitions the total observed variation into two main sources: the variation from the parts being measured and the variation from the measurement system. The measurement system variation is further broken down into **Repeatability** (equipment variation) and **Reproducibility** (operator variation).
    
    **Application:** This is the first and most critical gate in an assay transfer. You cannot validate a process with an unreliable measurement system. Before our Hero can fight the Villain of Process Variation, they must first prove their own weapon—the assay—is sharp, true, and reliable.
    """)
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig, pct_rr, pct_part = plot_gage_rr()
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: % Gage R&R", value=f"{pct_rr:.1f}%", delta="Lower is better", delta_color="inverse")
            st.metric(label="💡 KPI: % Part Variation", value=f"{pct_part:.1f}%", delta="Higher is better")
            st.markdown("- **Repeatability:** Inherent precision of the instrument/assay.")
            st.markdown("- **Reproducibility:** Variation between different operators.")
            st.markdown("**The Bottom Line:** A low % Gage R&R (<10%) proves that your measurement system is a reliable 'ruler' and that most of the variation you see in your process is real process variation, not measurement noise.")
        with tab2:
            st.markdown("Based on AIAG (Automotive Industry Action Group) guidelines:")
            st.markdown("- **< 10%:** System is **acceptable**.")
            st.markdown("- **10% - 30%:** **Conditionally acceptable**, depending on the importance of the application and cost of improvement.")
            st.markdown("- **> 30%:** System is **unacceptable** and requires improvement.")
        with tab3:
            st.markdown("""
            #### Origin and Development

            The concepts of Repeatability and Reproducibility have been a cornerstone of measurement science for over a century, but they were formally codified and popularized by the **Automotive Industry Action Group (AIAG)** in the 1980s. As part of the major quality revolution in the US auto industry, the AIAG created the Measurement Systems Analysis (MSA) manual, which established the Gage R&R study as a global standard for assessing the quality of a measurement system.

            The earliest methods for calculating Gage R&R were simple range-based calculations. However, these methods had a critical flaw: they could not separate the variation due to operator-part interaction from the variation due to the operators themselves.

            To solve this, the industry adopted **Analysis of Variance (ANOVA)** as the preferred method for Gage R&R studies. ANOVA, a technique pioneered by Sir Ronald A. Fisher, is a powerful statistical tool that can rigorously partition the total variation into its distinct sources: parts, operators, operator-part interaction, and the measurement equipment itself. This allows for a much more precise and insightful analysis of the measurement system's performance.

            ---
            
            #### Mathematical Basis

            The ANOVA method for Gage R&R is based on partitioning the total sum of squares ($SS_T$) into components attributable to each source of variation:
            """)
            st.latex(r"SS_T = SS_{Part} + SS_{Operator} + SS_{Interaction} + SS_{Error}")
            st.markdown("""
            From the Mean Squares (MS = SS/df) in the ANOVA table, we can estimate the variance components for each source:
            - **Repeatability (Equipment Variation, EV):** This is the inherent, random error of the measurement system itself, estimated directly from the Mean Square Error of the model.
            """)
            st.latex(r"\hat{\sigma}^2_{EV} = MS_{Error}")
            st.markdown("- **Reproducibility (Appraiser Variation, AV):** This is the variation introduced by different operators. It includes the main effect of the operator and the operator-part interaction.")
            st.latex(r"\hat{\sigma}^2_{AV} = \frac{MS_{Operator} - MS_{Interaction}}{n_{parts} \cdot n_{replicates}} + \frac{MS_{Interaction} - MS_{Error}}{n_{replicates}}")
            st.markdown("""
            The total **Gage R&R** variance is the sum of these two components:
            """)
            st.latex(r"\hat{\sigma}^2_{R\&R} = \hat{\sigma}^2_{EV} + \hat{\sigma}^2_{AV}")
            st.markdown("""
            The key KPI, **% Contribution**, is then calculated by comparing the Gage R&R variance to the total variation observed in the study:
            """)
            st.latex(r"\%R\&R = 100 \times \left( \frac{\hat{\sigma}^2_{R\&R}}{\hat{\sigma}^2_{Total}} \right)")
            
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
            st.markdown("""
            #### Origin and Development

            The mathematical foundation for this analysis is **Ordinary Least Squares (OLS) Regression**, a fundamental statistical method developed independently by Adrien-Marie Legendre (1805) and Carl Friedrich Gauss (1809). Gauss, working in the field of astronomy, developed the method to predict the orbits of celestial bodies from a limited number of observations. He was trying to find the "best fit" curve to describe the path of the dwarf planet Ceres.

            The core principle of OLS is to find the line that minimizes the sum of the squared vertical distances (the "residuals") between the observed data points and the fitted line. This concept of minimizing squared error is one of the most powerful and widely used ideas in all of statistics and machine learning.

            In the context of assay validation, we apply this centuries-old technique to answer a very modern question: "Does my instrument's response have a linear relationship with the true concentration of the substance I'm measuring?" It's a testament to the enduring power of the method that the same mathematics used to track planets is now used to validate life-saving medicines.

            ---
            
            #### Mathematical Basis

            We fit a simple linear model to the calibration data:
            """)
            st.latex("y = \\beta_0 + \\beta_1 x + \\epsilon")
            st.markdown("""
            - $y$ is the measured concentration (the response).
            - $x$ is the nominal (true) concentration.
            - $\\beta_0$ is the y-intercept, which represents the constant systematic bias of the assay (ideally 0).
            - $\\beta_1$ is the slope, which represents the proportional bias of the assay (ideally 1).
            - $\\epsilon$ is the random error term.

            The analysis then involves statistical tests on the estimated coefficients:
            - **Hypothesis Test for Slope:** $H_0: \\beta_1 = 1$ vs. $H_a: \\beta_1 \\neq 1$
            - **Hypothesis Test for Intercept:** $H_0: \\beta_0 = 0$ vs. $H_a: \\beta_0 \\neq 0$

            **Recovery** is a simple, direct measure of accuracy at each point:
            """)
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
            st.markdown("""
            #### Origin and Development

            The concepts of LOD and LOQ were formalized and harmonized for the pharmaceutical industry by the **International Council for Harmonisation (ICH)** in their **Q2(R1) guideline on Validation of Analytical Procedures**. Before ICH, different regulatory bodies had varying definitions and methods, leading to confusion and inconsistency. The ICH guidelines provided a scientifically sound and globally accepted framework for determining and validating these crucial performance characteristics. This ensures that an assay validated in one country will be accepted by regulators in another.

            The ICH Q2(R1) guideline describes several methods for determining LOD and LOQ, including:
            1.  **Based on Visual Evaluation:** Only applicable for non-instrumental methods.
            2.  **Based on Signal-to-Noise Ratio:** Typically used for methods that exhibit baseline noise, like chromatography.
            3.  **Based on the Standard Deviation of the Response and the Slope:** This is the most common and statistically robust approach for quantitative assays, and it is the method visualized in this toolkit.

            ---
            
            #### Mathematical Basis

            The most common method is based on the **standard deviation of the response ($\sigma$)** and the **slope of the calibration curve (S)**. The standard deviation of the response can be determined from the standard deviation of blank measurements or, more robustly, from the standard deviation of the residuals (or y-intercepts) from a low-level regression line.

            - **Limit of Detection (LOD):** The formula is derived to provide a high level of confidence that a signal at this level is not just random noise. The factor 3.3 is an approximation that corresponds to roughly 3 times the standard deviation of the noise.
            """)
            st.latex(r"LOD = \frac{3.3 \times \sigma}{S}")
            st.markdown("""
            - **Limit of Quantitation (LOQ):** This requires a higher signal-to-noise ratio to ensure not just detection, but also acceptable precision. The factor of 10 is the standard convention for this, providing a signal that is roughly 10 times the noise level.
            """)
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
            st.markdown("- **Deming Regression:** Checks for systematic constant (intercept) and proportional (slope) errors.")
            st.markdown("- **Bland-Altman Plot:** Visualizes the random error and quantifies the expected range of disagreement (Limits of Agreement).")
            st.markdown("- **% Bias Plot:** Directly assesses practical significance. Does the bias at any concentration exceed the pre-defined limits?")
            st.markdown("**The Bottom Line:** Passing all three analyses proves that the receiving lab's method is statistically indistinguishable from the reference method, confirming a successful transfer.")
        with tab2:
            st.markdown("- **Deming Regression:** The 95% confidence interval for the **slope should contain 1.0**, and the 95% CI for the **intercept should contain 0**.")
            st.markdown(f"- **Bland-Altman:** Greater than 95% of the data points must fall within the Limits of Agreement (`{la:.2f}` to `{ua:.2f}`). The width of this interval must be practically or clinically acceptable.")
            st.markdown("- **Percent Bias:** The bias at each concentration level should not exceed a pre-defined limit, often **±15%**. ")
        with tab3:
            st.markdown("""
            #### Origin and Development

            **The Problem with Simple Regression:** For decades, scientists incorrectly used standard Ordinary Least Squares (OLS) regression and correlation (R²) to compare methods. This approach is fundamentally flawed because it assumes the reference method (x-axis) is measured without error, which is never true. This leads to biased estimates of the slope and intercept.

            - **Deming's Solution:** **W. Edwards Deming**, a giant in the field of quality, popularized a more appropriate technique known as **Errors-in-Variables Regression** (often called Deming Regression). This method acknowledges that *both* the reference and test methods have inherent measurement error. It finds a line that minimizes the sum of squared errors in both the x and y directions, providing a much more accurate and unbiased estimate of the true relationship between the two methods.

            **The Problem with Correlation:** A high correlation (e.g., R² = 0.99) does not mean two methods agree. It only means that they are proportional to each other. For example, if one method consistently gives a result that is exactly double the other, the correlation would be perfect, but the methods clearly do not agree.

            - **The Bland-Altman Revolution:** In a landmark 1986 paper in *The Lancet*, statisticians **J. Martin Bland and Douglas G. Altman** addressed this widespread misuse of correlation. They proposed a simple, intuitive graphical method that directly assesses **agreement**. By plotting the *difference* between the two methods against their *average*, their plot makes it easy to visualize the mean bias, the scatter around the bias (random error), and any trends in the bias across the measurement range. It has since become the gold standard for method comparison studies in the clinical and life sciences.

            ---
            
            #### Mathematical Basis

            **Deming Regression:**
            Unlike OLS, which minimizes vertical distances to the line, Deming regression minimizes the sum of squared perpendicular distances from the data points to the regression line, weighted by the ratio of the error variances ($\lambda = \sigma^2_y / \sigma^2_x$). The slope ($\beta_1$) is calculated as:
            """)
            st.latex(r"\beta_1 = \frac{(S_{yy} - \lambda S_{xx}) + \sqrt{(S_{yy} - \lambda S_{xx})^2 + 4\lambda S_{xy}^2}}{2S_{xy}}")
            st.markdown(r"""
            Where $S_{xx}$, $S_{yy}$, and $S_{xy}$ are the sums of squares and cross-products.

            **Bland-Altman Plot:**
            For each sample $i$, calculate:
            - Average: $(\frac{x_i+y_i}{2})$
            - Difference: $(y_i - x_i)$
            
            The plot shows the Difference versus the Average. The key metrics are the **mean bias** ($\bar{d}$) and the **Limits of Agreement (LoA)**, which define the range where 95% of future differences are expected to lie:
            """)
            st.latex(r"LoA = \bar{d} \pm 1.96 \cdot s_d")
            st.markdown("where $s_d$ is the standard deviation of the differences.")
            
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
            st.markdown("""
            #### Origin and Development

            **The Agricultural Revolution (DOE):** The foundation of modern DOE was single-handedly pioneered by the legendary British statistician **Sir Ronald A. Fisher** in the 1920s and 1930s. Working at the Rothamsted Agricultural Experimental Station, he was faced with a complex problem: how to test the effect of multiple factors (like fertilizer type, seed variety, watering schedules) on crop yield when the underlying experimental material (plots of land) was inherently variable. His solution was revolutionary: instead of testing one factor at a time, he developed **factorial designs** to test multiple factors simultaneously in a structured way. This not only saved immense time and resources but, crucially, was the only way to systematically study **interactions**—the way the effect of one factor changes depending on the level of another. His work, published in his seminal 1935 book "The Design of Experiments," laid the groundwork for a century of scientific and industrial progress.

            **The Chemical Revolution (RSM):** In the 1950s, working in the chemical industry at Imperial Chemical Industries (ICI), statisticians **George E. P. Box and K. B. Wilson** built upon Fisher's work to solve a new problem: not just identifying which factors were important, but finding the *optimal settings* for those factors to maximize a response (like chemical yield). They developed **Response Surface Methodology (RSM)**, a sequential approach that uses an initial screening DOE to identify key factors, followed by a more detailed "optimizing" design (like the Central Composite Design shown in the plots) to model the curvature in the response. This allowed them to mathematically "map" the process and find the peak of the mountain, a task that was previously done through expensive and inefficient trial and error.

            ---
            
            #### Mathematical Basis

            **DOE (Screening):**
            A 2-level factorial design is used to estimate the main effects of factors and their interactions by fitting a linear model. For two factors, Temperature ($X_1$) and pH ($X_2$), the model is:
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2 + \epsilon")
            st.markdown("""
            Where the coefficients ($\beta$) are calculated from the experimental results and represent the standardized effects. The Pareto plot is a bar chart of these effects, which are tested for statistical significance using ANOVA to separate real effects from random noise.

            **RSM (Optimization):**
            After identifying the critical factors, RSM is used to model the curvature of the response surface. This requires adding center and axial points to the design to allow for the estimation of quadratic terms. The model is expanded to a second-order polynomial:
            """)
            st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{11} X_1^2 + \beta_{22} X_2^2 + \beta_{12} X_1 X_2 + \epsilon")
            st.markdown("""
            This equation describes the 3D surface shown in the plot. By taking the partial derivatives of this equation with respect to each factor and setting them to zero, we can mathematically solve for the exact settings ($X_1, X_2$) that correspond to the maximum (or minimum) response, thereby finding the process's optimal point.
            """)

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
            st.markdown("""
            **Origin:** Developed by Walter A. Shewhart at Bell Labs in the 1920s, these charts are the foundation of modern Statistical Process Control (SPC). Shewhart's breakthrough was recognizing that industrial processes contain two types of variation: common cause (the natural, inherent "noise" of a stable process) and special cause (unexpected, external events). The purpose of a Shewhart chart is not to eliminate all variation, but to provide a clear, statistical signal to distinguish between these two types, allowing engineers and scientists to focus their efforts on fixing real problems (special causes) instead of chasing random noise.
            
            **Mathematical Basis:** The key is estimating the process standard deviation ($\hat{\sigma}$) from the average moving range ($\overline{MR}$).
            """)
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
            st.markdown("""
            #### Origin and Development

            **Shewhart's Limitation:** The foundational Shewhart chart is a powerful tool, but its design is optimized for detecting *large* shifts in the process mean (typically 1.5σ or greater). It achieves this by treating each data point independently, giving it no "memory" of past performance. This makes it relatively insensitive to small, sustained shifts where the process mean might drift by only 0.5σ or 1σ. In this scenario, individual points are unlikely to fall outside the ±3σ limits, yet the process is clearly no longer centered on its target.

            **The Need for Memory:** Both EWMA and CUSUM were developed in the 1950s to address this specific limitation by incorporating "memory" into the control chart.

            - **EWMA (Exponentially Weighted Moving Average):** Proposed by S. W. Roberts in a 1959 paper, this method creates a smoothed average where the influence of past data points decays exponentially. This averaging effect filters out random noise, making the underlying trend or small shift much easier to detect. It's akin to looking at a 7-day rolling average of stock prices instead of the chaotic daily price.

            - **CUSUM (Cumulative Sum):** Developed by E. S. Page in 1954, this method is even more direct. It literally accumulates the deviations of each data point from the target. If the process is on target, the positive and negative deviations will cancel out, and the cumulative sum will hover around zero. If the process shifts even slightly, the deviations will consistently be in one direction, causing the cumulative sum to trend steadily up or down until it crosses a decision threshold. CUSUM is considered the most statistically powerful method for detecting small, sustained shifts of a specific magnitude.

            ---
            
            #### Mathematical Basis

            **EWMA:**
            The core of the EWMA chart is its recursive formula. The EWMA statistic at time $i$, denoted $z_i$, is a weighted average of the current observation $x_i$ and the previous EWMA value $z_{i-1}$:
            """)
            st.latex(r"z_i = \lambda x_i + (1-\lambda)z_{i-1}")
            st.markdown(r"""
            Where $\lambda$ (lambda) is the smoothing parameter ($0 < \lambda \le 1$). The control limits for EWMA are time-varying because the variance of the statistic decreases as more points are added:
            """)
            st.latex(r"UCL_i = \mu_0 + L \sigma \sqrt{\frac{\lambda}{2-\lambda} [1 - (1-\lambda)^{2i}]}")
            st.markdown(r"""
            Where $L$ is the number of standard deviations (typically 3), $\mu_0$ is the target mean, and $\sigma$ is the process standard deviation.

            **CUSUM:**
            The CUSUM chart uses two one-sided statistics to detect upward ($SH$) and downward ($SL$) shifts.
            - **Slack Value (k):** This is a reference value, also called the "allowance." It's the amount of deviation from the target that we are willing to tolerate before the sum starts to accumulate. It is typically set to half the magnitude of the shift ($\delta$) you want to detect. For example, to detect a 1-sigma shift ($\delta=1\sigma$), you would set $k = 0.5\sigma$.
            - **Decision Interval (H):** This is the control limit. When the cumulative sum exceeds this value, the process is considered out of control. It is typically set to 4 or 5 times the process standard deviation ($H=4\sigma$ or $H=5\sigma$).

            The recursive formulas are:
            """)
            st.latex(r"SH_i = \max(0, SH_{i-1} + (x_i - \mu_0) - k)")
            st.latex(r"SL_i = \max(0, SL_{i-1} + (\mu_0 - x_i) - k)")
            st.markdown(r"""
            The `max(0, ...)` term is crucial; it prevents the sum from "recovering." If the process drifts back towards the mean, the sum simply resets to zero, ready to detect the next shift. A signal is triggered if $SH_i > H$ or $SL_i > H$.
            """)

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
            st.markdown("""
            #### Origin and Development
            
            **From Industry to the Clinic:** The concept of control charts originated with Walter Shewhart in industrial manufacturing in the 1920s. In the 1950s, **Dr. S. Levey and Dr. E. R. Jennings** adapted this powerful tool for the clinical laboratory, creating what is now known as the Levey-Jennings chart. This was a revolutionary step in bringing statistical quality control to healthcare.
            
            **The Westgard Revolution:** For decades, labs often used simple ±2σ or ±3σ limits. However, **Dr. James Westgard** recognized that this approach was a blunt instrument—either too prone to false alarms (2s) or not sensitive enough to real problems (3s). In a landmark 1981 paper, he proposed a "multi-rule" system. This system combines several different rules to create a highly sensitive yet specific quality control procedure. The Westgard Rules are now the global standard for clinical laboratory QC and are essential for meeting regulatory requirements from bodies like CLIA, CAP, and ISO 15189.
            
            ---
            
            #### Mathematical Basis
            The rules are based on the probabilities of points from a normal distribution falling within specific sigma zones. Patterns that are highly improbable under normal conditions are flagged as signals.
            """)
            
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

elif "Process Capability (Cpk)" in method_key:
    st.markdown("""
    **Purpose:** To determine if a stable process is capable of consistently producing results that meet the required specifications.
    
    **Definition:** Process Capability is a measure of the ability of a process to produce output within the specification limits. The Cpk index is a standard metric that quantifies this ability.
    
    **Application:** This is often the final, critical gate of a successful assay transfer. After our Hero has proven their assay is reliable (Act I) and stable in the new lab (Act II), they must now provide the ultimate proof: that the process can consistently defeat the Villain of Variation and deliver results that meet the non-negotiable quality targets. A high Cpk is the statistical equivalent of a "mission accomplished."
    """)
    scenario = st.sidebar.radio("Select Process Scenario:", ('Ideal', 'Shifted', 'Variable', 'Out of Control'))
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        fig, cpk_val, scn = plot_capability(scenario)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Analysis & Interpretation")
        tab1, tab2, tab3 = st.tabs(["💡 Key Insights", "✅ Acceptance Rules", "📖 Method Theory"])
        with tab1:
            st.metric(label="📈 KPI: Process Capability (Cpk)", value=f"{cpk_val:.2f}" if scn != 'Out of Control' else "INVALID")
            st.markdown("- **The Mantra:** Control before Capability. Cpk is only meaningful for a stable, in-control process (see I-Chart in the plot).")
            st.markdown("- **The 'Holy Shit' Moment:** A process can be perfectly **in control but not capable** (the 'Shifted' and 'Variable' scenarios). The control chart looks fine, but the process is producing scrap. This is why you need both tools.")
        with tab2:
            st.markdown("- `Cpk ≥ 1.33`: Process is considered **capable** (a common minimum target).")
            st.markdown("- `Cpk ≥ 1.67`: Process is considered **highly capable** (a common Six Sigma target).")
            st.markdown("- `Cpk < 1.0`: Process is **not capable** of meeting specifications.")
        with tab3:
            st.markdown("""
            #### Origin and Development

            The concept of process capability indices originated in the manufacturing industry, particularly in Japan in the 1970s, as part of the Total Quality Management (TQM) revolution. However, it was the rise of **Six Sigma** at Motorola in the 1980s that truly popularized Cpk and made it a global standard for quality.

            The core idea of Six Sigma is to reduce process variation so that the nearest specification limit is at least six standard deviations away from the process mean. A Cpk of 2.0 corresponds to a true Six Sigma process.

            The framework provides a simple, standardized language to communicate the performance of a process relative to its requirements (the "voice of the customer"). It answers the crucial business question: "Is our process good enough to meet our customers' needs?"

            ---
            
            #### Mathematical Basis

            Capability analysis compares the **"Voice of the Process"** (the actual spread of the data, typically defined as a 6σ spread) to the **"Voice of the Customer"** (the allowable spread defined by the specification limits).

            - **Cp (Potential Capability):** Measures if the process is *narrow enough* to fit within the specification limits, but it does not account for centering.
            """)
            st.latex(r"C_p = \frac{USL - LSL}{6\hat{\sigma}}")
            st.markdown("""
            - **Cpk (Actual Capability):** This is the more important metric. It measures if the process is narrow enough *and* well-centered. It is the lesser of the upper and lower capability indices, effectively measuring the distance from the process mean to the *nearest* specification limit.
            """)
            st.latex(r"C_{pk} = \min(C_{pu}, C_{pl}) = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right)")
            st.markdown("""
            A Cpk of 1.0 means the process 3σ spread exactly fits within half of the specification range, with the mean touching the nearest limit. A Cpk of 1.33 means there is a "buffer" equivalent to one standard deviation between the process edge and the nearest specification limit.
            """)

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
            st.markdown("""
            #### Origin and Development

            The **Isolation Forest** algorithm was proposed by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in a groundbreaking 2008 paper. It represented a fundamental shift in how to think about anomaly detection.

            Previous methods were often "density-based" or "distance-based." They tried to define what a "normal" region looks like and then flagged anything that fell outside of it. This approach can be computationally expensive and struggles in high-dimensional spaces (the "curse of dimensionality").

            The authors of Isolation Forest flipped the problem on its head. They started with a simple but powerful observation: **anomalies are "few and different."** Because they are different, they should be easier to *isolate* from the rest of the data points. Instead of trying to describe the dense crowd of normal points, they built a method that was explicitly designed to find the lonely outliers. This counter-intuitive approach proved to be both highly effective and computationally efficient, making it one of the most popular and powerful unsupervised anomaly detection algorithms in use today.

            ---
            
            #### Mathematical Basis

            The core idea is that if you randomly partition a dataset, anomalies will be isolated in fewer steps than normal points. The algorithm works as follows:
            1.  An ensemble of "Isolation Trees" (iTrees) is built.
            2.  To build an iTree, the data is recursively partitioned by randomly selecting a feature and then randomly selecting a split point between the min and max values of that feature.
            3.  This continues until every point is isolated in its own leaf node. Anomalous points, being different, will require fewer partitions and will therefore have a much shorter average path length from the root to the leaf.
            
            The anomaly score $s(x, n)$ for an instance $x$ is derived from its average path length $E(h(x))$ across all the trees in the forest:
            """)
            st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")
            st.markdown("""
            - Where $c(n)$ is a normalization factor based on the average path length in a Binary Search Tree.
            - Scores close to 1 indicate a very short path length and are therefore flagged as **anomalies**.
            - Scores much smaller than 0.5 indicate a long path length and are considered **normal**.
            """)

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
            st.markdown("""
            #### Origin and Development

            **Logistic regression** is a statistical model developed by the brilliant British statistician **Sir David Cox in 1958**. Its origins lie in the need to model the probability of a binary event (e.g., a patient surviving or not, a component failing or not) as a function of one or more predictor variables.

            While linear regression predicts a continuous value that can go to positive or negative infinity, this doesn't make sense for probabilities, which must be constrained between 0 and 1. Cox's breakthrough was to use the **logistic function (or sigmoid function)** to "squash" the output of a linear equation into this [0, 1] range.

            The method's power lies in its **interpretability**. Unlike more complex "black box" models, the coefficients of a fitted logistic regression model have a direct and understandable meaning: they represent the change in the log-odds of the outcome for a one-unit change in the predictor variable. This makes it a foundational and still widely used algorithm in medicine (for predicting disease risk), finance (for predicting loan default), and, as shown here, in industrial quality control.

            ---
            
            #### Mathematical Basis

            The model works in two steps:
            1.  First, a linear combination of the input features ($x$) is created, identical to linear regression. This is often called the **log-odds** or **logit**.
            """)
            st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n")
            st.markdown("""
            2.  Second, this result is passed through the sigmoid function, $\sigma(z)$, to transform the log-odds into a probability between 0 and 1.
            """)
            st.latex(r"P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown("""
            The model is trained by finding the optimal coefficients ($\beta_0, \beta_1, ...$) that maximize the likelihood of observing the training data. The decision boundary visualized in the plot is the line (or surface) where the predicted probability is exactly 0.5 (i.e., where $z=0$).
            """)

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
            st.markdown("""
            #### Origin and Development

            **Prophet** is an open-source forecasting procedure developed and released by **Facebook's Core Data Science team in 2017**. It was created to address a common and difficult business problem: producing high-quality forecasts at scale, often with minimal manual effort from analysts.

            Traditional forecasting methods like ARIMA or exponential smoothing are powerful but often require deep statistical knowledge, careful parameter tuning, and stationary data (data with a constant mean and variance). Business and scientific time series data, however, are rarely so well-behaved. They often have:
            - Strong, multiple seasonalities (e.g., weekly, yearly)
            - Shifting trends (e.g., a process slowly degrading)
            - Missing data and outliers
            - The effects of known, irregular events (e.g., holidays, maintenance shutdowns)

            Prophet was designed from the ground up to handle these features automatically. It frames the forecasting problem as a curve-fitting exercise, making it intuitive for analysts to understand and tune. Its ability to automatically detect trend changepoints and model multiple seasonalities makes it an exceptionally robust and practical tool for real-world data, including the kind of control data generated in a lab.

            ---
            
            #### Mathematical Basis

            Prophet is a **decomposable time series model**, which means it models the time series as a combination of several distinct components:
            """)
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("""
            - **$g(t)$ is the trend function.** Prophet models the trend using either a saturating logistic growth model or a simpler piecewise linear model. A key feature is its ability to automatically detect "changepoints"—points in time where the trend's growth rate changes significantly.

            - **$s(t)$ is the seasonality function.** This component models periodic changes in the data (e.g., yearly patterns in assay performance due to ambient temperature changes). Prophet models seasonality using a flexible **Fourier series**, which allows it to fit complex seasonal patterns.

            - **$h(t)$ is the holidays/events function.** This component models the effects of known, potentially irregular events with a specified window of influence (e.g., instrument maintenance, a known change in a reagent lot).

            - **$\epsilon_t$ is the error term,** which is assumed to be normally distributed noise that is not captured by the other components.

            The entire model is fit within a Bayesian framework, which allows Prophet to produce uncertainty intervals for the forecast and to incorporate prior beliefs about the parameters if desired.
            """)

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
            st.markdown("""
            #### Origin and Development

            **The Flawed "Standard":** For many years, the most commonly taught method for calculating a confidence interval for a proportion was the **Wald interval**. It is based on a simple normal approximation to the binomial distribution. While easy to calculate, this method was known to have terrible performance, especially for small sample sizes or when the observed proportion was close to 0 or 1. It often produces intervals that are nonsensically narrow or that extend beyond the possible range of [0, 1].

            **The Search for a Better Way:** Recognizing these flaws, statisticians developed more robust methods in the early 20th century.
            - **Wilson Score Interval (1927):** Developed by American statistician Edwin Bidwell Wilson, this method is also based on a normal approximation, but it inverts the more robust "score test" rather than the Wald test. This seemingly small change has a dramatic effect on its performance. It produces reasonable intervals even in extreme cases and is now widely recommended as the best general-purpose interval for proportions.
            - **Clopper-Pearson Interval (1934):** Developed by C.J. Clopper and E.S. Pearson, this is known as an "exact" method because it is based directly on the cumulative binomial distribution. It is constructed by inverting two one-sided binomial tests. The result is an interval that is guaranteed to have a coverage probability of *at least* the nominal level (e.g., 95%). While this guarantee is powerful, it often makes the interval overly conservative (i.e., wider than necessary).

            ---
            
            #### Mathematical Basis

            Let $\hat{p} = k/n$ be the observed proportion of $k$ successes in $n$ trials, and let $z$ be the Z-score for the desired confidence (e.g., 1.96 for 95%).
            
            - **Wald Interval:**
            """)
            st.latex(r"\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            st.markdown("""
            This simple formula is the source of its problems, as the standard error term approaches zero when $\hat{p}$ is near 0 or 1.
            
            - **Wilson Score Interval:** It is the solution to the quadratic equation that arises from inverting the score test, which leads to the more complex but far superior formula:
            """)
            st.latex(r"\frac{1}{1 + z^2/n} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right)")
            st.markdown("""
            Notice how it "shrinks" the observed proportion $\hat{p}$ towards 0.5 by adding the term $z^2/2n$. This is a form of regularization that improves its performance.

            - **Clopper-Pearson (Exact) Interval:** It is defined using the Beta distribution, which is directly related to the cumulative binomial distribution.
            - **Lower Bound:** $B(\alpha/2; k, n-k+1)$
            - **Upper Bound:** $B(1-\alpha/2; k+1, n-k)$
            Where $B$ is the quantile function (or inverse CDF) of the Beta distribution.
            """)
            
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
            st.markdown("""
            #### Origin and Development

            The fundamental theorem that underpins all of Bayesian statistics was conceived by the **Reverend Thomas Bayes**, an English statistician and philosopher, in the 1740s. It was published posthumously in 1763. For nearly two centuries, however, Bayes' Theorem remained largely a theoretical curiosity. The reason was simple: for all but the most trivial problems, the mathematics required to calculate the posterior distribution were intractable.

            This all changed with the advent of powerful computers. The "Bayesian revolution" began in the late 20th century with the development of **modern computational techniques** like **Markov Chain Monte Carlo (MCMC)**. These algorithms allow computers to approximate the posterior distribution through simulation, even for highly complex models that are impossible to solve analytically.

            This computational power transformed Bayesian statistics from a niche philosophical viewpoint into one of the most powerful and flexible frameworks for data analysis, now widely used in fields from astrophysics to drug development and artificial intelligence.

            ---
            
            #### Mathematical Basis

            The relationship between the components is defined by **Bayes' Theorem**:
            """)
            st.latex(r"P(\theta | \text{Data}) = \frac{P(\text{Data} | \theta) P(\theta)}{P(\text{Data})}")
            st.markdown("""
            In practice, the denominator, $P(\text{Data})$, is a normalizing constant that is often difficult to calculate. So, we often work with the proportional form, which captures the shape of the posterior distribution:
            """)
            st.latex(r"\underbrace{P(\theta | \text{Data})}_{\text{Posterior}} \propto \underbrace{P(\text{Data} | \theta)}_{\text{Likelihood}} \times \underbrace{P(\theta)}_{\text{Prior}}")
            st.markdown("""
            For binomial data (like pass/fail rates), the **Beta distribution** is a **conjugate prior** for the binomial likelihood. This is a special mathematical relationship that makes the calculation simple. It means that if you start with a Beta prior and get binomial data, your posterior will also be a Beta distribution.
            - If Prior is Beta($\\alpha, \\beta$) and Data is $k$ successes in $n$ trials:
            - The Posterior is Beta($\\alpha + k, \\beta + n - k$).
            """)
            
elif "Confidence Interval Concept" in method_key:
    st.markdown("""
    **Purpose:** To understand the fundamental concept and correct interpretation of frequentist confidence intervals.
    
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
            st.markdown("- **Theoretical Universe (Top Plot):** This shows why inference is possible. The wide blue curve is the true population. The narrow orange curve is the distribution of *all possible sample means*. Because it's so narrow, any single sample mean we draw is very likely to be close to the true population mean.")
            st.markdown("- **CI Simulation (Bottom Plot):** As you increase `n` with the slider, the confidence intervals become dramatically shorter, reflecting the increased precision shown in the top plot.")
            st.markdown("- **The 'Holy Shit' Moment (Diminishing Returns):** Notice that the gain in precision (the narrowing of the orange curve and the shortening of the CIs) from n=5 to n=20 is huge. The gain from n=80 to n=100 is much smaller. This demonstrates the law of diminishing returns in sampling, a critical concept for designing efficient experiments.")
        with tab2:
            st.markdown("This is a teaching module, not a validation step. The 'acceptance rule' is to **correctly interpret the CI** in all reports and discussions:")
            st.error("🔴 **Incorrect:** 'There is a 95% probability that the true mean is in this interval.'")
            st.success("🟢 **Correct:** 'We are 95% confident that this interval contains the true mean.' This is shorthand for: 'This interval was constructed using a procedure that, in the long run, captures the true mean 95% of the time.'")
        with tab3:
            st.markdown("""
            #### Origin and Development

            The concept of **confidence intervals** was introduced by the brilliant Polish mathematician and statistician **Jerzy Neyman** in a landmark 1937 paper. At the time, the field of statistics was dominated by a fierce debate between the Bayesian approach and the "fiducial inference" approach of Sir Ronald A. Fisher.

            Neyman sought a third way. He wanted a method that was rigorously frequentist—meaning its properties could be defined by long-run frequencies of repeated experiments—but that also provided a practical, intuitive range of plausible values for a parameter.

            His solution was the confidence interval. It was a revolutionary idea: instead of trying to make a probabilistic statement about the fixed, unknown parameter, Neyman made a probabilistic statement about the *procedure* used to create the interval. He proved that one could construct an interval in such a way that, over many repeated experiments, a certain percentage (e.g., 95%) of those intervals would capture the true parameter.

            This elegant and practical solution was a huge success. The Neyman-Pearson framework, which includes both confidence intervals and hypothesis testing, quickly became the dominant paradigm in applied statistics and remains the foundation of statistical inference in most scientific fields today.

            ---
            
            #### Mathematical Basis

            The general form of a two-sided confidence interval is:
            """)
            st.latex(r"\text{CI} = \text{Point Estimate} \pm (\text{Margin of Error})")
            st.markdown("""
            Where the Margin of Error is defined as:
            """)
            st.latex(r"(\text{Critical Value}) \times (\text{Standard Error of the Point Estimate})")
            st.markdown("""
            - **Point Estimate:** Our best guess for the parameter from our sample (e.g., the sample mean, $\\bar{x}$).
            - **Standard Error:** The standard deviation of the sampling distribution of the point estimate. For the mean, it's $\frac{s}{\sqrt{n}}$.
            - **Critical Value:** A value from a probability distribution (like the t-distribution or Z-distribution) that corresponds to the desired level of confidence. For a 95% CI, it's the value that cuts off the top and bottom 2.5% of the distribution.

            For the mean of a population with an unknown standard deviation (the most common case), this becomes:
            """)
            st.latex(r"\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}")

