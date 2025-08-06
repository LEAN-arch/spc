import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import beta
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from prophet import Prophet

# ==============================================================================
# APP CONFIGURATION
# ==============================================================================
st.set_page_config(layout="wide", page_title="Authoritative Assay Transfer Toolkit")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stExpander { border: 1px solid #e6e6e6; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def cohen_d(x, y):
    """Calculate Cohen's d for independent samples"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def clopper_pearson_interval(successes, n, alpha=0.05):
    """Calculate the Clopper-Pearson exact interval."""
    if n == 0: return (0, 1)
    lower = stats.beta.ppf(alpha/2, successes, n - successes + 1)
    upper = stats.beta.ppf(1 - alpha/2, successes + 1, n - successes)
    return (lower if not np.isnan(lower) else 0, upper if not np.isnan(upper) else 1)

def wilson_score_interval(p_hat, n, z=1.96):
    """Calculates the Wilson score interval for a binomial proportion."""
    if n == 0: return (0, 1)
    term1 = (p_hat + z**2 / (2 * n)); denom = 1 + z**2 / n
    term2 = z * np.sqrt((p_hat * (1-p_hat)/n) + (z**2 / (4 * n**2)))
    return (term1 - term2) / denom, (term1 + term2) / denom

# ==============================================================================
# PLOTTING FUNCTIONS (All 15 Methods, Assay Transfer Theme)
# ==============================================================================

def plot_gage_rr():
    """Gage R&R to quantify measurement system error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle('Gage R&R Study: Quantifying Measurement System Error', fontweight='bold', fontsize=16)
    np.random.seed(10); n_operators, n_samples, n_replicates = 3, 10, 3
    sample_means = np.linspace(90, 110, n_samples); operator_bias = [0, -0.5, 0.8]; data = []
    for op_idx, operator in enumerate(['Alice', 'Bob', 'Charlie']):
        for sample_idx, sample_mean in enumerate(sample_means):
            measurements = np.random.normal(sample_mean + operator_bias[op_idx], 1.5, n_replicates)
            for m in measurements: data.append([operator, f'Sample_{sample_idx+1}', m])
    df = pd.DataFrame(data, columns=['Operator', 'Sample', 'Measurement'])
    model = ols('Measurement ~ C(Sample) + C(Operator) + C(Sample):C(Operator)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    ms_operator = anova_table['sum_sq']['C(Operator)'] / anova_table['df']['C(Operator)']
    ms_interaction = anova_table['sum_sq']['C(Sample):C(Operator)'] / anova_table['df']['C(Sample):C(Operator)']
    ms_error = anova_table['sum_sq']['Residual'] / anova_table['df']['Residual']
    var_repeatability = ms_error
    var_reproducibility = ((ms_operator - ms_interaction) / (n_samples * n_replicates)) + ((ms_interaction - ms_error) / n_replicates)
    var_part = (anova_table['sum_sq']['C(Sample)'] / anova_table['df']['C(Sample)'] - ms_interaction) / (n_operators * n_replicates)
    variances = {k: max(0, v) for k, v in locals().items() if 'var_' in k}
    var_rr = variances['var_repeatability'] + variances['var_reproducibility']
    var_total = var_rr + variances['var_part']
    pct_rr = (var_rr / var_total) * 100 if var_total > 0 else 0
    pct_part = (variances['var_part'] / var_total) * 100 if var_total > 0 else 0
    
    sns.boxplot(x='Sample', y='Measurement', data=df, ax=ax1, color='lightgray', showfliers=False)
    sns.stripplot(x='Sample', y='Measurement', data=df, hue='Operator', ax=ax1, jitter=True, dodge=True, palette='viridis')
    ax1.set_title('Measurements by Sample and Operator'); ax1.tick_params(axis='x', rotation=45)
    ax2.bar(['% Gage R&R', '% Part-to-Part'], [pct_rr, pct_part], color=['salmon', 'skyblue'])
    ax2.set_ylabel('Percent of Total Variation'); ax2.set_title('Variance Contribution')
    ax2.axhline(10, color='darkgreen', linestyle='--', label='<10% (Acceptable)'); ax2.axhline(30, color='darkorange', linestyle='--', label='>30% (Unacceptable)')
    ax2.legend(); ax2.text(0, pct_rr, f'{pct_rr:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold'); ax2.text(1, pct_part, f'{pct_part:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_linearity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)); fig.suptitle('Assay Linearity and Range Verification', fontweight='bold', fontsize=16); np.random.seed(42)
    nominal = np.array([10, 25, 50, 100, 150, 200, 250]); measured = nominal + np.random.normal(0, 2, len(nominal)) - (nominal/150)**3
    X = sm.add_constant(nominal); model = sm.OLS(measured, X).fit(); b, m = model.params; residuals = model.resid
    ax1.plot(nominal, measured, 'o', label='Measured Values', markersize=8); ax1.plot(nominal, model.predict(X), 'r-', label=f'Best Fit Line (y={m:.3f}x + {b:.2f})'); ax1.plot([0, 260], [0, 260], 'k--', label='Line of Identity (y=x)')
    ax1.set_xlabel('Nominal Concentration (ng/mL)'); ax1.set_ylabel('Measured Concentration (ng/mL)'); ax1.set_title('Linearity Plot'); ax1.legend(); ax1.grid(True)
    ax2.plot(nominal, residuals, 'go'); ax2.axhline(0, color='k', linestyle='--'); ax2.set_xlabel('Nominal Concentration (ng/mL)'); ax2.set_ylabel('Residuals (Measured - Predicted)'); ax2.set_title('Residual Analysis'); ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig, model

def plot_lod_loq():
    fig, ax = plt.subplots(); np.random.seed(3); blanks = np.random.normal(1.5, 0.5, 20); low_conc = np.random.normal(5.0, 0.6, 20)
    mean_blank, std_blank = np.mean(blanks), np.std(blanks, ddof=1); LOD = mean_blank + 3.3 * std_blank; LOQ = mean_blank + 10 * std_blank
    sns.kdeplot(blanks, ax=ax, fill=True, label='Blank Sample Distribution'); sns.kdeplot(low_conc, ax=ax, fill=True, label='Low Concentration Sample Distribution')
    ax.axvline(LOD, color='orange', linestyle='--', lw=2, label=f'LOD = {LOD:.2f}'); ax.axvline(LOQ, color='red', linestyle='--', lw=2, label=f'LOQ = {LOQ:.2f}')
    ax.set_title('Limit of Detection (LOD) and Quantitation (LOQ)', fontweight='bold'); ax.set_xlabel('Assay Signal (e.g., Absorbance)'); ax.set_ylabel('Density'); ax.legend(); return fig

def plot_method_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)); fig.suptitle('Method Comparison: R&D Lab vs QC Lab', fontweight='bold', fontsize=16); np.random.seed(42); x = np.linspace(20, 150, 50); y = 0.98 * x + 1.5 + np.random.normal(0, 2.5, 50)
    delta = np.var(y, ddof=1) / np.var(x, ddof=1); x_mean, y_mean = np.mean(x), np.mean(y); Sxx = np.sum((x - x_mean)**2); Syy = np.sum((y - y_mean)**2); Sxy = np.sum((x - x_mean)*(y - y_mean))
    beta1_deming = (Syy - delta*Sxx + np.sqrt((Syy - delta*Sxx)**2 + 4*delta*Sxy**2)) / (2*Sxy); beta0_deming = y_mean - beta1_deming*x_mean
    avg = (x + y) / 2; diff = y - x; mean_diff = np.mean(diff); std_diff = np.std(diff, ddof=1); upper_loa = mean_diff + 1.96 * std_diff; lower_loa = mean_diff - 1.96 * std_diff
    ax1.plot(x, y, 'o', label='Sample Results'); ax1.plot(x, beta0_deming + beta1_deming*x, 'r-', label=f'Deming Fit (y={beta1_deming:.2f}x + {beta0_deming:.2f})'); ax1.plot([0, 160], [0, 160], 'k--', label='Line of Identity'); ax1.set_xlabel('R&D Lab Measurement (Reference)'); ax1.set_ylabel('QC Lab Measurement (Test)'); ax1.set_title('Deming Regression'); ax1.legend(); ax1.grid(True)
    ax2.plot(avg, diff, 'o', color='purple'); ax2.axhline(mean_diff, color='red', linestyle='-', label=f'Mean Bias = {mean_diff:.2f}'); ax2.axhline(upper_loa, color='blue', linestyle='--', label=f'Upper LoA = {upper_loa:.2f}'); ax2.axhline(lower_loa, color='blue', linestyle='--', label=f'Lower LoA = {lower_loa:.2f}'); ax2.set_xlabel('Average of Methods'); ax2.set_ylabel('Difference (QC - R&D)'); ax2.set_title('Bland-Altman Agreement Plot'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

def plot_robustness():
    fig = plt.figure(figsize=(12,8)); data = {'Temp': [-1, 1, -1, 1, -1, 1, -1, 1], 'pH': [-1, -1, 1, 1, -1, -1, 1, 1], 'Time': [-1, -1, -1, -1, 1, 1, 1, 1]}; df = pd.DataFrame(data)
    df['Response'] = 100 + 5*df['Temp'] - 2*df['pH'] + 1.5*df['Time'] - 3*df['Temp']*df['pH'] + np.random.normal(0, 1, 8); model = ols('Response ~ Temp * pH * Time', data=df).fit(); effects = model.params.iloc[1:]; effects = effects.sort_values(key=abs, ascending=True)
    ax1 = fig.add_subplot(2,1,1); effects.plot(kind='barh', ax=ax1); ax1.set_title('Assay Robustness: Pareto Plot of Standardized Effects', fontweight='bold'); ax1.set_xlabel('Effect Magnitude on Assay Response')
    ax2 = fig.add_subplot(2,2,3); sns.pointplot(data=df, x='Temp', y='Response', hue='pH', ax=ax2, markers=['o', 's'], linestyles=['-', '--']); ax2.set_title('Significant Interaction: Temp * pH'); ax2.set_xticklabels(['Low Temp', 'High Temp']); plt.tight_layout(); return fig

def plot_shewhart():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]}); fig.suptitle('Assay Stability Monitoring: I-MR Chart of Daily Controls', fontweight='bold', fontsize=16); np.random.seed(42); in_control = np.random.normal(100.0, 2.0, 15); reagent_shift = np.random.normal(108.0, 2.0, 10); data = np.concatenate([in_control, reagent_shift]); x = np.arange(1, len(data) + 1); mean = np.mean(data[:15]); mr = np.abs(np.diff(data)); mr_mean = np.mean(mr[:14]); sigma_est = mr_mean / 1.128; UCL_I, LCL_I = mean + 3 * sigma_est, mean - 3 * sigma_est; out_of_control_I = np.where((data > UCL_I) | (data < LCL_I))[0]; UCL_MR = mr_mean * 3.267
    ax1.plot(x, data, 'o-', c='royalblue', label='Control Sample Result'); ax1.axhline(mean, c='black', ls='--', label=f'Established Mean ({mean:.1f} ng/mL)'); ax1.axhline(UCL_I, c='red', ls='-', label=f'UCL={UCL_I:.1f}'); ax1.axhline(LCL_I, c='red', ls='-'); ax1.scatter(x[out_of_control_I], data[out_of_control_I], c='red', s=150, zorder=5, label='Mean Shift Signal'); ax1.axvspan(15.5, 25.5, color='orange', alpha=0.2, label='New Reagent Lot Introduced'); ax1.set_ylabel('Concentration (ng/mL)'); ax1.set_title('I-Chart: Monitors Assay Accuracy (Bias)'); ax1.legend(loc='upper left')
    ax2.plot(x[1:], mr, 'o-', c='teal', label='Moving Range (Run-to-Run)'); ax2.axhline(mr_mean, c='black', ls='--', label=f'Avg. Range ({mr_mean:.1f})'); ax2.axhline(UCL_MR, c='red', ls='-', label=f'UCL={UCL_MR:.1f}'); ax2.set_ylabel('Range (ng/mL)'); ax2.set_xlabel('Analytical Run Number'); ax2.set_title('MR-Chart: Monitors Assay Precision (Variability)'); ax2.legend(loc='upper left'); plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

def plot_ewma_cusum(chart_type, lmbda, k_sigma, H_sigma):
    np.random.seed(101); data = np.concatenate([np.random.normal(50, 2, 25), np.random.normal(52.5, 2, 15)]); target = np.mean(data[:25]); sigma = np.std(data[:25], ddof=1)
    if chart_type == 'EWMA':
        fig, ax = plt.subplots(); ewma_vals = np.zeros_like(data); ewma_vals[0] = target
        for i in range(1, len(data)): ewma_vals[i] = lmbda * data[i] + (1 - lmbda) * ewma_vals[i-1]
        L = 3; UCL = [target + L*sigma*np.sqrt((lmbda/(2-lmbda))*(1-(1-lmbda)**(2*i))) for i in range(1, len(data)+1)]; out_idx = np.where(ewma_vals > UCL)[0]
        ax.plot(data, 'o-', c='gray', alpha=0.4, label='Daily Control Value', markersize=4); ax.plot(ewma_vals, 'o-', c='purple', label=f'EWMA (λ={lmbda})'); ax.plot(UCL, c='red', ls='-', label='EWMA UCL'); ax.set_ylabel('Assay Response (e.g., EU/mL)')
        if len(out_idx) > 0: ax.scatter(out_idx[0], ewma_vals[out_idx[0]], c='red', s=150, zorder=5, label=f'Signal')
        ax.set_title(f'EWMA Chart for Detecting Slow Assay Drift', fontweight='bold')
    else:
        fig, ax = plt.subplots(); k = k_sigma * sigma; H = H_sigma * sigma; SH, SL = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)): SH[i] = max(0, SH[i-1] + (data[i] - target) - k); SL[i] = max(0, SL[i-1] + (target - data[i]) - k)
        out_idx = np.where((SH > H) | (SL > H))[0]
        ax.plot(SH, 'o-', c='darkcyan', label='High-Side CUSUM (SH)'); ax.plot(SL, 'o-', c='darkorange', label='Low-Side CUSUM (SL)'); ax.axhline(H, color='red', linestyle='-', label=f'Control Limit H={H:.1f}'); ax.set_ylabel('Cumulative Sum')
        if len(out_idx) > 0: ax.scatter(out_idx[0], SH[out_idx[0]], c='red', s=150, zorder=5, label='Signal')
        ax.set_title(f'CUSUM Chart for Detecting Sustained Shifts', fontweight='bold')
    ax.axvspan(25, 40, color='orange', alpha=0.2, label='Small (1.25σ) Process Shift'); ax.axhline(target if chart_type=='EWMA' else 0, c='black', ls='--', label='Target'); ax.legend(loc='upper left'); ax.set_xlabel('Analytical Run Number'); plt.tight_layout(); return fig

def plot_multi_rule(rule_set='Westgard'):
    fig, ax = plt.subplots(); np.random.seed(3); mean, std = 100, 2; data = np.concatenate([np.random.normal(mean, std, 5), [mean + 2.1*std, mean + 2.2*std], np.random.normal(mean, std, 2), np.linspace(mean - 0.5*std, mean - 2*std, 6), [mean + 3.5*std], np.random.normal(mean + 1.5*std, 0.3, 4), np.random.normal(mean, std, 3), np.random.normal(mean - 1.5*std, 0.3, 5)]); x = np.arange(1, len(data) + 1); ax.fill_between(x, mean+2*std, mean+3*std, color='orange', alpha=0.2, label='±2σ to ±3σ Zone (Warning)'); ax.fill_between(x, mean-3*std, mean-2*std, color='orange', alpha=0.2); ax.fill_between(x, mean+std, mean+2*std, color='gold', alpha=0.2, label='±1σ to ±2σ Zone'); ax.fill_between(x, mean-2*std, mean-std, color='gold', alpha=0.2); ax.axhline(mean, c='black', lw=1.5, ls='--', label=f'Nominal Value = {mean}')
    for i in [-3, -2, -1, 1, 2, 3]: ax.axhline(mean+i*std, c='gray', lw=1, ls=':'); ax.text(len(data)+0.5, mean+i*std, f'{i}σ', va='center', ha='left', fontsize=10)
    ax.plot(x, data, 'o-', c='darkblue', markersize=5, label='QC Sample Results'); title = f'{rule_set} Multi-Rule Chart for Analytical Run Validation'
    if rule_set == 'Westgard': ax.annotate('1_3s Violation\n(Run REJECTED)', xy=(14, data[13]), xytext=(10, 108), arrowprops=dict(facecolor='red', shrink=0.05), c='red', weight='bold'); ax.annotate('2_2s Violation\n(Run REJECTED)', xy=(7, data[6]), xytext=(3, 108), arrowprops=dict(facecolor='red', shrink=0.05), c='red', weight='bold'); ax.annotate('4_1s Violation\n(Systematic Error)', xy=(16, data[15]), xytext=(18, 108), arrowprops=dict(facecolor='orange', shrink=0.05), c='orange', weight='bold')
    else: title = 'Nelson Rules for Manufacturing Process Control'; ax.annotate('Rule 1\n(Process Halt)', xy=(14, data[13]), xytext=(10, 108), arrowprops=dict(facecolor='red', shrink=0.05), c='red', weight='bold'); ax.annotate('Rule 3\n(Trend Detected)', xy=(13, data[12]), xytext=(9, 92), arrowprops=dict(facecolor='orange', shrink=0.05), c='orange', weight='bold')
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('QC Run Number'); ax.set_ylabel('Measured Value'); ax.legend(loc='lower left'); ax.set_xlim(0, len(data) + 2); plt.tight_layout(); return fig

def plot_capability(Cpk_target):
    fig, ax = plt.subplots(); np.random.seed(42); LSL, USL = 90, 110; process_mean = 101; process_std = (USL - LSL) / (6 * Cpk_target); data = np.random.normal(process_mean, process_std, 200); sigma_hat = np.std(data, ddof=1); Cpu = (USL - data.mean()) / (3 * sigma_hat); Cpl = (data.mean() - LSL) / (3 * sigma_hat); Cpk = np.min([Cpu, Cpl])
    sns.histplot(data, ax=ax, kde=True, stat='density', label='Process Data'); ax.axvline(LSL, color='red', linestyle='--', lw=2.5, label=f'LSL = {LSL}'); ax.axvline(USL, color='red', linestyle='--', lw=2.5, label=f'USL = {USL}'); ax.axvline(data.mean(), color='black', linestyle=':', lw=2, label=f'Process Mean = {data.mean():.2f}'); ax.set_title('Process Capability Analysis (Cpk)', fontweight='bold', fontsize=16); ax.legend()
    status = "Pass" if Cpk >= 1.33 else "Fail"; color = "darkgreen" if status == "Pass" else "darkred"; info_text = f'Cpk = {Cpk:.2f}\nTarget >= 1.33\nStatus: {status}'; ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.7), color='white'); plt.tight_layout(); return fig

def plot_anomaly_detection():
    fig, ax = plt.subplots(figsize=(10,8)); np.random.seed(42); X_normal = np.random.multivariate_normal([100, 20], [[5, 2],[2, 1]], 200); X_anomalies = np.array([[95, 25], [110, 18], [115, 28]]); X = np.vstack([X_normal, X_anomalies]); model = IsolationForest(n_estimators=100, contamination=0.015, random_state=42); model.fit(X); y_pred = model.predict(X)
    DecisionBoundaryDisplay.from_estimator(model, X, ax=ax, response_method="predict", cmap='Blues_r', alpha=0.4); scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm_r', s=50, edgecolor='k'); legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=['Normal Run', 'Anomaly'], title="Status"); ax.add_artist(legend1)
    ax.set_title('Multivariate Anomaly Detection (Isolation Forest)', fontweight='bold'); ax.set_xlabel('Assay Response (e.g., Fluorescence Units)'); ax.set_ylabel('Incubation Time (min)'); return fig

def plot_predictive_qc():
    fig, ax = plt.subplots(figsize=(10, 8)); np.random.seed(1); n_points = 150; X1 = np.random.normal(5, 2, n_points); X2 = np.random.normal(25, 3, n_points); logit_p = -15 + 1.5 * X1 + 0.5 * X2 + np.random.normal(0, 2, n_points); p = 1 / (1 + np.exp(-logit_p)); y = np.random.binomial(1, p); X = np.vstack([X1, X2]).T
    model = LogisticRegression().fit(X, y); DecisionBoundaryDisplay.from_estimator(model, X, ax=ax, response_method="predict_proba", pcolormesh_kw={'alpha': 0.4, 'cmap':'RdYlGn_r'}, xlabel='Reagent Age (days)', ylabel='Incubation Temp (°C)')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn', edgecolor='k'); ax.legend(handles=scatter.legend_elements()[0], labels=['Pass', 'Fail'], title="Actual Outcome"); ax.set_title('Predictive QC: Predicting Run Failure with Logistic Regression', fontweight='bold'); return fig

def plot_forecasting():
    np.random.seed(42); dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W')); trend = np.linspace(0, 5, 104); seasonality = 1.5 * np.sin(np.arange(104) * (2 * np.pi / 52.14)); noise = np.random.normal(0, 0.5, 104); y = 50 + trend + seasonality + noise; df = pd.DataFrame({'ds': dates, 'y': y}); model = Prophet(weekly_seasonality=False, daily_seasonality=False); model.fit(df); future = model.make_future_dataframe(periods=26, freq='W'); forecast = model.predict(future)
    fig1 = model.plot(forecast, xlabel='Date', ylabel='Control Value (U/mL)'); ax = fig1.gca(); ax.set_title('Time Series Forecasting of Control Performance', fontweight='bold'); ax.axhline(58, c='red', ls='--', label='Upper Spec Limit'); ax.legend()
    fig2 = model.plot_components(forecast); return fig1, fig2

def plot_wilson(successes, n_samples):
    fig, ax = plt.subplots(); p_hat = successes / n_samples if n_samples > 0 else 0; wald_lower, wald_upper = stats.norm.interval(0.95, loc=p_hat, scale=np.sqrt(p_hat*(1-p_hat)/n_samples)) if n_samples > 0 else (0,0); wilson_lower, wilson_upper = wilson_score_interval(p_hat, n_samples); cp_lower, cp_upper = clopper_pearson_interval(successes, n_samples)
    intervals = {"Wald (Approximate)": (wald_lower, wald_upper, 'red'), "Wilson Score": (wilson_lower, wilson_upper, 'blue'), "Clopper-Pearson (Exact)": (cp_lower, cp_upper, 'green')};
    for i, (name, (lower, upper, color)) in enumerate(intervals.items()): ax.plot([lower, upper], [i, i], lw=8, color=color, solid_capstyle='round', label=f"{name}: [{lower:.3f}, {upper:.3f}]")
    ax.axvline(p_hat, c='black', ls='--', label=f'Observed Rate p̂={p_hat:.3f}'); ax.set_yticks(range(len(intervals))); ax.set_yticklabels(intervals.keys()); ax.set_xlabel('Concordance Rate'); ax.set_title(f'Comparing CIs for {successes}/{n_samples} Concordant Results', fontweight='bold'); ax.legend(loc='lower right'); ax.set_xlim(-0.05, 1.05); plt.tight_layout(); return fig

def plot_bayesian(prior_type):
    fig, ax = plt.subplots(); n_qc, successes_qc = 20, 18; observed_rate = successes_qc / n_qc
    if prior_type == "No Prior (Frequentist)": prior_alpha, prior_beta = 1, 1
    elif prior_type == "Strong R&D Prior": prior_alpha, prior_beta = 490, 10
    else: prior_alpha, prior_beta = 10, 10
    prior_dist = beta.pdf(np.linspace(0, 1, 1000), prior_alpha, prior_beta); posterior_alpha, posterior_beta = prior_alpha + successes_qc, prior_beta + (n_qc - successes_qc); posterior_dist = beta.pdf(np.linspace(0, 1, 1000), posterior_alpha, posterior_beta); cred_interval = beta.interval(0.95, posterior_alpha, posterior_beta)
    ax.plot(np.linspace(0,1,1000), posterior_dist, 'b-', lw=3, label=f'Posterior Belief'); ax.fill_between(np.linspace(0,1,1000), posterior_dist, where=(np.linspace(0,1,1000) >= cred_interval[0]) & (np.linspace(0,1,1000) <= cred_interval[1]), color='blue', alpha=0.2, label=f'95% Credible Interval\n[{cred_interval[0]:.2f}, {cred_interval[1]:.2f}]'); ax.plot(np.linspace(0,1,1000), prior_dist, 'g--', lw=2, label=f'Prior Belief ({prior_type})'); ax.axvline(observed_rate, color='red', linestyle=':', lw=2, label=f'QC Lab Data ({observed_rate:.2f})')
    ax.set_title('Bayesian Inference for Assay Concordance Rate', fontweight='bold'); ax.set_xlabel('Assay Pass Rate (Concordance)'); ax.set_ylabel('Probability Density'); ax.legend(loc='upper left'); ax.set_xlim(0.6, 1.0); plt.tight_layout(); return fig

def plot_ci_concept():
    fig, ax = plt.subplots(); np.random.seed(123); pop_mean, pop_std, n = 100, 15, 30; n_sims = 100; capture_count = 0
    for i in range(n_sims):
        sample = np.random.normal(pop_mean, pop_std, n); sample_mean = np.mean(sample); margin_of_error = 1.96 * (pop_std / np.sqrt(n)); ci_lower, ci_upper = sample_mean - margin_of_error, sample_mean + margin_of_error
        color = 'cornflowerblue' if ci_lower <= pop_mean <= ci_upper else 'red'
        if color == 'cornflowerblue': capture_count += 1
        ax.plot([ci_lower, ci_upper], [i, i], color=color, lw=2); ax.plot(sample_mean, i, 'o', color='black', markersize=3)
    capture_rate = capture_count / n_sims; ax.axvline(pop_mean, color='black', linestyle='--', lw=2, label=f'True Population Mean (μ={pop_mean})'); ax.set_title(f'Conceptual Simulation of 100 95% Confidence Intervals', fontweight='bold'); ax.set_xlabel('Value'); ax.set_ylabel('Simulation Run'); ax.legend(loc='upper right')
    info_text = (f"Out of 100 simulations, {capture_count} CIs ({capture_rate:.0%}) 'captured' the true mean.\nThis demonstrates the meaning of 95% confidence: the *procedure* works\n95% of the time, not that any single interval has a 95% probability."); ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9)); ax.set_ylim(-2, n_sims + 2); plt.tight_layout(); return fig

# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
st.title("🔬 The Authoritative Assay Transfer Statistical Toolkit")
st.markdown("A comprehensive, interactive guide bridging **Classical SPC** and **Modern ML/AI** for a successful technical assay transfer and lifecycle management.")

PLOT_METHODS = {
    "1. Gage R&R (Measurement System Analysis)": "gage_rr",
    "2. Linearity and Range": "linearity",
    "3. LOD & LOQ (Assay Sensitivity)": "lod_loq",
    "4. Method Comparison (Deming/Bland-Altman)": "method_comp",
    "5. Assay Robustness (DOE)": "robustness",
    "6. Process Stability (Shewhart I-MR)": "shewhart",
    "7. Small Shift Detection (EWMA/CUSUM)": "ewma_cusum",
    "8. Run Validation (Westgard/Nelson)": "multi_rule",
    "9. Process Capability (Cpk Analysis)": "capability",
    "10. Advanced Anomaly Detection (ML)": "anomaly_detection",
    "11. Predictive Quality Control (ML)": "predictive_qc",
    "12. Control Forecasting (Time Series AI)": "forecasting",
    "13. Pass/Fail Assay Analysis (Wilson Score)": "wilson",
    "14. Leveraging Historical Data (Bayesian)": "bayesian",
    "15. Confidence Intervals (The Concept)":"ci_concept"
}

st.sidebar.title("⚙️ Assay Transfer & Lifecycle Protocol")
selection_key = st.sidebar.selectbox("Select a Validation Step:", options=list(PLOT_METHODS.keys()))
selection = PLOT_METHODS[selection_key]

st.header(f"Step: {selection_key}")

# --- Method Display Logic ---
if selection == 'gage_rr':
    st.pyplot(plot_gage_rr())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** Before measuring the process, you must measure your measurement system. A Gage R&R study quantifies the inherent variability (error) of the assay itself.
*   **Acceptance Rules (AIAG Guidelines):**
    *   **% Gage R&R < 10%:** The measurement system is acceptable.
    *   **% Gage R&R 10% - 30%:** May be acceptable based on the importance of the application and cost of improvement.
    *   **% Gage R&R > 30%:** The measurement system is unacceptable and requires improvement. This is a critical first gate in any assay transfer.""")
    with st.expander("Method Theory: Gage R&R and ANOVA"):
        st.markdown("""**Origin:** Gage R&R studies became a cornerstone of the automotive industry's quality initiatives and are formalized by the Automotive Industry Action Group (AIAG). The use of ANOVA to partition the sources of variation is the statistically rigorous and preferred method.
        **Mathematical Basis:** The total sum of squares ($SS_T$) is partitioned: $ SS_T = SS_{Part} + SS_{Operator} + SS_{Interaction} + SS_{Error} $. From the Mean Squares (MS), we estimate variance components:
        *   **Repeatability (EV):** $\hat{\sigma}^2_{EV} = MS_{Error}$
        *   **Reproducibility (AV):** $\hat{\sigma}^2_{AV} = \frac{MS_{Operator} - MS_{Interaction}}{n_{parts} \cdot n_{replicates}}$
        *   **Gage R&R:** $\hat{\sigma}^2_{R\&R} = \hat{\sigma}^2_{EV} + \hat{\sigma}^2_{AV}$
        The study variation is then calculated: $ \%R\&R = 100 \times \left( \frac{\hat{\sigma}_{R\&R}}{\hat{\sigma}_{Total}} \right) $""")

elif selection == 'linearity':
    fig, model = plot_linearity()
    st.pyplot(fig)
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown(f"""**Objective:** To verify the assay's ability to provide results that are directly proportional to the concentration of the analyte over a specified range.
*   **Acceptance Rules:**
    *   **Coefficient of Determination (R²):** Should typically be **> 0.995**. (This fit: **R² = {model.rsquared:.4f}**)
    *   **Slope:** Should be close to 1.0 (e.g., within **0.95 - 1.05**). (This fit: **Slope = {model.params[1]:.3f}**)
    *   **Intercept:** The 95% CI for the intercept should contain 0. (This fit: **Intercept = {model.params[0]:.2f}**)
    *   **Residual Plot:** Should show a random scatter of points around the zero line with no obvious patterns.""")
    with st.expander("Method Theory: Linearity by Regression"):
        st.markdown("""**Origin:** Based on Ordinary Least Squares (OLS) regression, a fundamental statistical method developed by Legendre and Gauss in the early 1800s. In assay validation, we test the hypothesis that the measured values have a linear relationship with the nominal (true) values.
        **Mathematical Basis:** We fit the model $y = \beta_0 + \beta_1 x + \epsilon$.
        *   $y$ is the measured concentration.
        *   $x$ is the nominal concentration.
        *   $\beta_1$ is the slope (ideal = 1).
        *   $\beta_0$ is the y-intercept or constant bias (ideal = 0).
        *   $\epsilon$ is the random error.
        We test the null hypotheses $H_0: \beta_1 = 1$ and $H_0: \beta_0 = 0$. The R² value measures the proportion of the variance in the dependent variable that is predictable from the independent variable.""")

elif selection == 'lod_loq':
    st.pyplot(plot_lod_loq())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To determine the lowest concentrations at which the assay can reliably detect and quantify an analyte.
*   **Limit of Detection (LOD):** The lowest analyte concentration that can be reliably distinguished from a blank sample.
*   **Limit of Quantitation (LOQ):** The lowest analyte concentration that can be measured with an acceptable level of precision and accuracy.
*   **Acceptance Rules:** The experimentally determined **LOQ must be less than or equal to the lowest concentration that needs to be measured** for the assay's intended use.""")
    with st.expander("Method Theory: LOD & LOQ"):
        st.markdown("""**Origin:** Based on the recommendations from the International Council for Harmonisation (ICH) Q2(R1) guidelines, which provide a framework for analytical procedure validation.
        **Mathematical Basis (Signal-to-Noise approach):**
        This common method uses the standard deviation of the response of blank samples.
        *   **LOD:** Estimated as the mean of the blank plus 3.3 times the standard deviation of the blank.
            $$ LOD = \bar{y}_{blank} + 3.3 \sigma_{blank} $$
        *   **LOQ:** Estimated as the mean of the blank plus 10 times the standard deviation of the blank.
            $$ LOQ = \bar{y}_{blank} + 10 \sigma_{blank} $$
        The factors 3.3 and 10 are derived from the slope of the calibration curve and desired confidence levels, and have become standard conventions.""")
    
elif selection == 'method_comp':
    st.pyplot(plot_method_comparison())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To formally assess the agreement and bias between two methods (e.g., old vs. new, R&D vs. QC).
*   **Deming Regression:** Used when both methods have measurement error. It assesses for constant and proportional bias.
    *   **Acceptance Rule:** The **95% CI for the slope should contain 1.0** and the **95% CI for the intercept should contain 0**.
*   **Bland-Altman Plot:** Visualizes the agreement between the two methods.
    *   **Acceptance Rule:** The **mean bias should be close to zero**, and **>95% of the data points should fall within the calculated Limits of Agreement (LoA)**. The width of the LoA must be clinically or practically acceptable.""")
    with st.expander("Method Theory: Deming Regression & Bland-Altman"):
        st.markdown("""**Origin:**
        *   **Deming Regression:** Named after W. Edwards Deming, it's an errors-in-variables model that accounts for error in both x and y variables.
        *   **Bland-Altman Plot:** Introduced by J. Martin Bland and Douglas G. Altman in a 1986 Lancet paper.
        **Mathematical Basis:**
        *   **Deming:** Minimizes the sum of squared perpendicular distances from points to the line, weighted by the error variance ratio.
        *   **Bland-Altman:** Plots Difference $(y_i - x_i)$ vs. Average $(\frac{x_i+y_i}{2})$. Limits of Agreement are $\bar{d} \pm 1.96 \cdot s_d$. """)

elif selection == 'robustness':
    st.pyplot(plot_robustness())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To deliberately vary key assay parameters to identify which factors have a significant impact on the results.
*   **Pareto Plot:** Ranks the factors from most to least significant, focusing control efforts on the "vital few".
*   **Interaction Plot:** Visualizes how the effect of one factor changes at different levels of another.
*   **Acceptance Rules:** The goal is to **prove that small, expected variations in parameters do NOT significantly impact the assay result**. If a factor is significant, its operating range must be tightened in the SOP.""")
    with st.expander("Method Theory: Design of Experiments (DOE)"):
        st.markdown("""**Origin:** Pioneered by Sir Ronald A. Fisher in the 1920s. Factorial designs efficiently study multiple factors and their interactions.
        **Mathematical Basis:** A 2-level factorial design fits a linear model: $ y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{12} X_1 X_2 + ... $. The coefficients ($\beta$) represent the standardized effects, which are ranked in the Pareto plot.""")
        
elif selection == 'shewhart':
    st.pyplot(plot_shewhart())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** Before any comparison, the receiving lab must demonstrate it can run the assay in a stable, predictable manner.
*   **Acceptance Rules:** A process is considered stable and ready for the next validation step only when **at least 20-25 consecutive points on both the I-chart and MR-chart show no out-of-control signals**.""")
    with st.expander("Method Theory: Shewhart Charts"):
        st.markdown("""**Origin:** Developed by Walter A. Shewhart at Bell Labs in the 1920s.
        **Mathematical Basis:** Estimate process standard deviation via $\hat{\sigma} = \frac{\overline{MR}}{d_2}$ (where $d_2=1.128$).
        *   **I-Chart Limits:** $UCL/LCL = \bar{x} \pm 3\hat{\sigma}$
        *   **MR-Chart Limits:** $UCL = D_4 \overline{MR}$ (where $D_4=3.267$).""")

elif selection == 'ewma_cusum':
    chart_type = st.sidebar.radio("Select Chart Type:", ('EWMA', 'CUSUM'))
    lmbda, k_sigma, H_sigma = 0, 0, 0
    if chart_type == 'EWMA': lmbda = st.sidebar.slider("EWMA Lambda (λ)", 0.05, 1.0, 0.2, 0.05)
    else: k_sigma = st.sidebar.slider("CUSUM Slack (k, in σ)", 0.25, 1.5, 0.5, 0.25); H_sigma = st.sidebar.slider("CUSUM Limit (H, in σ)", 2.0, 8.0, 5.0, 0.5)
    st.pyplot(plot_ewma_cusum(chart_type, lmbda, k_sigma, H_sigma))
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To detect small, systematic drifts or shifts that a Shewhart chart might miss.
*   **EWMA:** Excellent for detecting small *drifts*. **Rule:** Use a small `λ` (0.1-0.3) for long-term monitoring.
*   **CUSUM:** The most efficient method for detecting a small, but *abrupt and sustained* shift. **Rule:** Set `k` to half the magnitude of the shift you want to detect.""")
    with st.expander("Method Theory: EWMA and CUSUM"):
        st.markdown("""**Origin:** EWMA by S. W. Roberts (1959). CUSUM by E. S. Page (1954).
        **Mathematical Basis:**
        *   **EWMA:** $ z_i = \lambda x_i + (1-\lambda)z_{i-1} $
        *   **CUSUM:** $SH_i = \max(0, SH_{i-1} + (x_i - \mu_0) - k)$. Signal if $SH_i > H$. """)

elif selection == 'multi_rule':
    rule_set = st.sidebar.radio("Select Rule Set:", ('Westgard', 'Nelson'))
    st.pyplot(plot_multi_rule(rule_set))
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To create an objective, statistically-driven system for accepting or rejecting each analytical run.
*   **Westgard Rules:** **Rule:** A run is rejected if a "rejection rule" like 1_3s, 2_2s, or R_4s is violated.
*   **Nelson Rules:** **Rule:** Any of the 8 rules being triggered indicates an out-of-control process.""")
    with st.expander("Method Theory: Westgard & Nelson Rules"):
        st.markdown("""**Origin:** Nelson Rules (1984) and Westgard Rules (1980s) adapt Shewhart charts to increase sensitivity to non-random patterns based on combinatorial probability.""")

elif selection == 'capability':
    Cpk_target = st.sidebar.slider("Simulate a process with Cpk target:", 0.5, 2.0, 1.33, 0.01)
    st.pyplot(plot_capability(Cpk_target))
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To determine if the process is capable of consistently producing results that meet specifications.
*   **Acceptance Rules:**
    *   **Cpk >= 1.33:** Process is considered capable (common minimum target).
    *   **Cpk >= 1.67:** Process is considered highly capable (Six Sigma target).
    *   **Cpk < 1.0:** Process is not capable.""")
    with st.expander("Method Theory: Process Capability (Cpk)"):
        st.markdown("""**Origin:** Developed in manufacturing as part of Six Sigma and TQM initiatives.
        **Mathematical Basis:** $ C_{pk} = \min \left( \frac{USL - \bar{x}}{3\hat{\sigma}}, \frac{\bar{x} - LSL}{3\hat{\sigma}} \right) $. It measures the distance to the nearest specification limit in units of 3-sigma.""")
        
elif selection == 'anomaly_detection':
    st.pyplot(plot_anomaly_detection())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To detect complex, multivariate anomalies that traditional control charts would miss.
*   **Rule:** This is an exploratory tool. Any data point flagged as an anomaly **must be investigated** by SMEs to determine the root cause.""")
    with st.expander("Method Theory: Isolation Forest"):
        st.markdown("""**Origin:** Proposed by Liu, Ting, and Zhou in 2008. An unsupervised algorithm based on the principle that anomalies are easier to isolate.
        **Mathematical Basis:** The anomaly score $ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $ is based on the average path length $E(h(x))$ to isolate a point. Shorter paths yield scores closer to 1 (anomaly).""")

elif selection == 'predictive_qc':
    st.pyplot(plot_predictive_qc())
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To move from reactive to proactive quality control by predicting run failure based on in-process parameters.
*   **Rule:** A risk threshold is established. For example: "If the predicted probability of failure is **> 20%**, flag the run for operator review before proceeding." """)
    with st.expander("Method Theory: Logistic Regression"):
        st.markdown("""**Origin:** Developed by statistician David Cox in 1958 for binary classification.
        **Mathematical Basis:** It models the probability of a binary outcome using the sigmoid function: $ P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ...)}} $.""")

elif selection == 'forecasting':
    fig1, fig2 = plot_forecasting()
    st.pyplot(fig1); st.pyplot(fig2)
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To forecast the future performance of assay controls to anticipate problems.
*   **Rule:** A "proactive alert" is triggered if the **lower bound of the 80% forecast interval (yhat_lower) is predicted to cross a specification limit** within the forecast horizon.""")
    with st.expander("Method Theory: Time Series Forecasting (Prophet)"):
        st.markdown("""**Origin:** An open-source forecasting procedure developed by Facebook.
        **Mathematical Basis:** A decomposable time series model: $ y(t) = g(t) + s(t) + h(t) + \epsilon_t $, where $g(t)$ is trend, $s(t)$ is seasonality, and $h(t)$ is holiday effects.""")

elif selection == 'wilson':
    n_samples = st.sidebar.slider("Number of Validation Samples (n)", 1, 100, 30)
    successes = st.sidebar.slider("Concordant Results", 0, n_samples, int(n_samples * 0.95))
    st.pyplot(plot_wilson(successes, n_samples))
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To validate qualitative or pass/fail assays by assessing their concordance rate.
*   **Acceptance Rule:** A common criterion is: "The **lower bound of the 95% Wilson Score (or Clopper-Pearson) confidence interval must be greater than or equal to the target concordance rate** (e.g., 90%)." """)
    with st.expander("Method Theory: Confidence Intervals for Proportions"):
        st.markdown("""**Origin:** The Wilson Score (1927) and Clopper-Pearson (1934) intervals were developed to provide much better performance than the standard Wald interval, especially for small samples.""")

elif selection == 'bayesian':
    prior_type = st.sidebar.selectbox("Select Prior:", ("Strong R&D Prior", "No Prior (Frequentist)", "Skeptical/Regulatory Prior"))
    st.pyplot(plot_bayesian(prior_type))
    with st.expander("SME Commentary & Acceptance Criteria"):
        st.markdown("""**Objective:** To use historical data (the Prior) to make a more informed decision with less new data.
*   **Acceptance Rule:** A Bayesian approach might specify: "The **95% credible interval for the concordance rate must be entirely above 90%**." """)
    with st.expander("Method Theory: Bayesian Statistics"):
        st.markdown("""**Origin:** Based on Bayes' Theorem (18th century).
        **Mathematical Basis:** $ \text{Posterior} \propto \text{Likelihood} \times \text{Prior} $. For binomial data, we use the Beta-Binomial conjugate model: if Prior is Beta($\alpha, \beta$) and Data is $k$ successes in $n$ trials, then the Posterior is Beta($\alpha + k, \beta + n - k$).""")

elif selection == 'ci_concept':
    st.pyplot(plot_ci_concept())
    with st.expander("SME Commentary"):
        st.markdown("""**Objective:** To understand the fundamental concept behind frequentist confidence intervals. This is a teaching tool, not a validation step.
*   **The Correct Interpretation:** A 95% confidence interval means that if we were to repeat the sampling process an infinite number of times, **95% of the calculated intervals would contain the true, unknown population parameter**. The confidence is in the *procedure*, not in any single interval.""")
    with st.expander("Method Theory: Confidence Intervals"):
        st.markdown("""**Origin:** Introduced by Jerzy Neyman in the 1930s.
        **Mathematical Basis:** $ \text{CI} = \text{Point Estimate} \pm (\text{Critical Value}) \times (\text{Standard Error}) $. For the mean with unknown $\sigma$, this is: $ \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}} $.""")
