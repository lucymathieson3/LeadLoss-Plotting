import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks
from sklearn.utils import resample

# Function to calculate KDE
def calculate_kde(data, x_range, bandwidth):
    """
    Calculate the KDE for a given dataset.
    
    Parameters:
    - data (ndarray): The dataset for which KDE is calculated.
    - x_range (ndarray): Range of x-values over which KDE is evaluated.
    - bandwidth (float): Bandwidth parameter for KDE.
    
    Returns:
    - kde_values (ndarray): The KDE values over x_range.
    """
    kde = gaussian_kde(data, bw_method=bandwidth)
    kde_values = kde(x_range)
    return kde_values

# Function to find optimal bandwidth
def find_optimal_bandwidth(data):
    """
    Use cross-validation to find the optimal bandwidth for KDE.
    
    Parameters:
    - data (pd.Series or ndarray): The input data for bandwidth optimization.
    
    Returns:
    - optimal_bandwidth (float): The optimal bandwidth for KDE.
    """
    bandwidths = np.linspace(0.01, 0.1, 20)
    data_array = data.to_numpy()  # Convert the pandas Series to a numpy array
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
    grid.fit(data_array[:, None])  # Apply the None indexing to the numpy array
    return grid.best_params_['bandwidth']


# Function to bootstrap peak uncertainties
def bootstrap_peak_uncertainty(data, kde_func, x_range, bandwidth, n_bootstrap=1000):
    """
    Estimate uncertainties in detected peaks using bootstrap resampling.
    
    Parameters:
    - data (ndarray): Input dataset.
    - kde_func (function): Function to calculate KDE.
    - x_range (ndarray): Range of x-values over which KDE is evaluated.
    - bandwidth (float): Bandwidth for KDE.
    - n_bootstrap (int): Number of bootstrap iterations.
    
    Returns:
    - peak_means (ndarray): Mean positions of peaks from bootstrap samples.
    - peak_stds (ndarray): Standard deviations of peak positions.
    """
    kde_values = kde_func(data, x_range, bandwidth)
    original_peaks, _ = find_peaks(kde_values)
    original_peak_positions = x_range[original_peaks]
    
    bootstrapped_peaks = {peak: [] for peak in original_peak_positions}
    
    for _ in range(n_bootstrap):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        kde_values_resampled = kde_func(resampled_data, x_range, bandwidth)
        resampled_peaks, _ = find_peaks(kde_values_resampled)
        resampled_peak_positions = x_range[resampled_peaks]
        
        for peak in original_peak_positions:
            if len(resampled_peak_positions) > 0:
                closest_peak = min(resampled_peak_positions, key=lambda x: abs(x - peak))
                bootstrapped_peaks[peak].append(closest_peak)
    
    peak_means = []
    peak_stds = []
    for peak, resampled_peaks in bootstrapped_peaks.items():
        peak_means.append(np.mean(resampled_peaks))
        peak_stds.append(np.std(resampled_peaks))
    
    return np.array(peak_means), np.array(peak_stds)

# Function to plot CDF for penalized vs. non-penalized results
def plot_cdf_penalization(df_penalized, df_non_penalized):
    """
    Plot the CDFs for penalized and non-penalized datasets.
    
    Parameters:
    - df_penalized (pd.DataFrame): Penalized dataset.
    - df_non_penalized (pd.DataFrame): Non-penalized dataset.
    """
    plt.figure(figsize=(10, 6))
    
    # Penalized
    sorted_data_penalized = sorted(df_penalized[df_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)'])
    cumulative_penalized = [i / len(sorted_data_penalized) for i in range(len(sorted_data_penalized))]
    plt.plot(sorted_data_penalized, cumulative_penalized, label='Penalized', color='magenta')
    
    # Non-penalized
    sorted_data_non_penalized = sorted(df_non_penalized[df_non_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)'])
    cumulative_non_penalized = [i / len(sorted_data_non_penalized) for i in range(len(sorted_data_non_penalized))]
    plt.plot(sorted_data_non_penalized, cumulative_non_penalized, label='Non-Penalized', color='lightseagreen')


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Function to plot histograms with KDE for penalized vs. non-penalized results, with bootstrapped peak uncertainties
def plot_histogram_with_kde_penalization(df_penalized, df_non_penalized, x_range, bin_size=50, n_bootstrap=100):
    """
    Plot histograms and KDEs for penalized and non-penalized datasets, 
    with bootstrapped peak uncertainties.
    
    Parameters:
    - df_penalized (pd.DataFrame): Penalized dataset.
    - df_non_penalized (pd.DataFrame): Non-penalized dataset.
    - x_range (ndarray): Range of x-values for KDE.
    - bin_size (float): Bin size for histograms.
    - n_bootstrap (int): Number of bootstrap iterations.
    """
    plt.figure(figsize=(10, 6))
    
    # Penalized
    filtered_data_penalized = df_penalized[df_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    bins = np.arange(min(filtered_data_penalized), max(filtered_data_penalized) + bin_size, bin_size)
    optimal_bandwidth_penalized = find_optimal_bandwidth(filtered_data_penalized)
    kde_penalized_values = calculate_kde(filtered_data_penalized, x_range, optimal_bandwidth_penalized)
    peak_means_penalized, peak_stds_penalized = bootstrap_peak_uncertainty(filtered_data_penalized, calculate_kde, x_range, optimal_bandwidth_penalized, n_bootstrap=n_bootstrap)
    
    plt.hist(filtered_data_penalized, bins=bins, alpha=0.7, label=None, density=True, color='lightgray', edgecolor='darkgrey')
    plt.plot(x_range, kde_penalized_values, label='KDE Penalized', color='magenta', linewidth = 2)
    for i, peak in enumerate(peak_means_penalized):
        plt.errorbar(peak, kde_penalized_values[np.argmin(np.abs(x_range - peak))], 
                     xerr=peak_stds_penalized[i], fmt='o', color='magenta', markersize=5)

    # Non-penalized
    filtered_data_non_penalized = df_non_penalized[df_non_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    optimal_bandwidth_non_penalized = find_optimal_bandwidth(filtered_data_non_penalized)
    kde_non_penalized_values = calculate_kde(filtered_data_non_penalized, x_range, optimal_bandwidth_non_penalized)
    peak_means_non_penalized, peak_stds_non_penalized = bootstrap_peak_uncertainty(filtered_data_non_penalized, calculate_kde, x_range, optimal_bandwidth_non_penalized, n_bootstrap=n_bootstrap)
    
    plt.hist(filtered_data_non_penalized, bins=bins, alpha=0.7, label=None, density=True, color='lightgray', edgecolor='darkgrey')
    plt.plot(x_range, kde_non_penalized_values, label='KDE Non-Penalized', color='lightseagreen', linewidth = 2)
    for i, peak in enumerate(peak_means_non_penalized):
        plt.errorbar(peak, kde_non_penalized_values[np.argmin(np.abs(x_range - peak))], 
                     xerr=peak_stds_non_penalized[i], fmt='o', color='lightseagreen', markersize=5)
    
    plt.xlabel('Pb loss age (Ma)')
    plt.xlim(100,1000)
    plt.ylabel('Density')
    plt.title('Overlaid Histograms and KDEs for Penalized vs. Non-Penalized (> 100 Ma)')
    plt.legend()
    plt.grid(False)

    # Add the CDF inset plot
    ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='upper right')

    # Penalized CDF
    sorted_data_penalized = np.sort(filtered_data_penalized)
    cumulative_penalized = np.arange(1, len(sorted_data_penalized) + 1) / len(sorted_data_penalized)
    ax_inset.plot(sorted_data_penalized, cumulative_penalized, label='Penalized', color='magenta', linewidth=1)

    # Non-Penalized CDF
    sorted_data_non_penalized = np.sort(filtered_data_non_penalized)
    cumulative_non_penalized = np.arange(1, len(sorted_data_non_penalized) + 1) / len(sorted_data_non_penalized)
    ax_inset.plot(sorted_data_non_penalized, cumulative_non_penalized, label='Non-Penalized', color='lightseagreen', linewidth=1)

    # Customize the inset plot
    ax_inset.set_xlabel('Pb loss age (Ma)', fontsize=7)
    ax_inset.set_ylabel('Cumulative Probability', fontsize=7)
    ax_inset.set_title('CDF', fontsize=7)
    ax_inset.tick_params(axis='both', which='major', labelsize=7)
    ax_inset.set_xlim(100, 800)
    ax_inset.legend(fontsize=6)
    
    plt.show()


# Function to perform K-S test for penalized vs. non-penalized results
def ks_test_penalization(df_penalized, df_non_penalized):
    filtered_data_penalized = df_penalized[df_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    filtered_data_non_penalized = df_non_penalized[df_non_penalized['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    stat, p_val = ks_2samp(filtered_data_penalized, filtered_data_non_penalized)
    return {'K-S Statistic': stat, 'p-value': p_val}

# Function to calculate variance, standard deviation, and confidence intervals for penalized vs. non-penalized results
def calculate_stats_penalization(df_penalized, df_non_penalized):
    stats = []
    
    for df, label in zip([df_penalized, df_non_penalized], ['Penalized', 'Non-Penalized']):
        filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
        variance = np.var(filtered_data)
        std_dev = np.std(filtered_data)
        conf_interval = np.percentile(filtered_data, [2.5, 97.5])
        
        stats.append({
            'Penalization': label,
            'Variance': variance,
            'Standard Deviation': std_dev,
            '95% Confidence Interval': conf_interval
        })
    
    return pd.DataFrame(stats)

# Load the penalized and non-penalized data files
file_non_penalized = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/Penalisation comparison/Mt_Isa_GroupI_v3_MC_output_1-1400_np.csv'
file_penalized = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/Penalisation comparison/Mt_Isa_GroupI_v3_MC_output_1-1400_10Ma.csv'

df_penalized = pd.read_csv(file_penalized)
df_non_penalized = pd.read_csv(file_non_penalized)

# Set the x_range for KDE calculations
x_range = np.linspace(100, 1000, 1000)

# Plot CDF for penalized vs non-penalized
plot_cdf_penalization(df_penalized, df_non_penalized)

# Plot histograms with KDEs and bootstrapped peaks for penalized vs non-penalized
plot_histogram_with_kde_penalization(df_penalized, df_non_penalized, x_range)

# Perform K-S test for penalized vs non-penalized
ks_test_results_penalization = ks_test_penalization(df_penalized, df_non_penalized)
print(ks_test_results_penalization)

# Calculate variance, standard deviation, and confidence intervals for penalized vs non-penalized
stats_penalization = calculate_stats_penalization(df_penalized, df_non_penalized)
print(stats_penalization)

# Optionally save the results to CSV files
# stats_penalization.to_csv('penalization_stats.csv', index=False)

