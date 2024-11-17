import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pandas as pd

# Define your KDE function with variable bandwidth
def calculate_kde(data, x_range, bandwidth):
    """
    Calculate the KDE for a given dataset.

    Parameters:
        data (array-like): Input data for which the KDE is calculated.
        x_range (array-like): Range of x-values over which to evaluate the KDE.
        bandwidth (float): Bandwidth for the KDE.

    Returns:
        kde_values (array): KDE values evaluated over the given x_range.
    """
    kde = gaussian_kde(data, bw_method=bandwidth)
    kde_values = kde(x_range)
    return kde_values

# Function to calculate peaks
def find_kde_peaks(kde_values, x_range, height=0.1):
    """
    Identify peaks in the KDE.

    Parameters:
        kde_values (array-like): KDE values over the x_range.
        x_range (array-like): x-values corresponding to the KDE values.
        height (float): Minimum height of a peak.

    Returns:
        peak_positions (array): Positions of the identified peaks.
        peak_heights (array): Heights of the identified peaks.
    """
    peaks, _ = find_peaks(kde_values, height=height)
    peak_positions = x_range[peaks]
    return peak_positions, kde_values[peaks]

# Function to bootstrap peak uncertainties
def bootstrap_peak_uncertainty(data, kde_func, x_range, bandwidth, n_bootstrap=1000):
    """
    Bootstrap the uncertainties in peak positions.

    Parameters:
        data (array-like): Original dataset for bootstrapping.
        kde_func (function): Function to calculate KDE.
        x_range (array-like): Range of x-values for KDE evaluation.
        bandwidth (float): Bandwidth for KDE.
        n_bootstrap (int): Number of bootstrap iterations.

    Returns:
        peak_means (array): Mean positions of bootstrapped peaks.
        peak_stds (array): Standard deviations of bootstrapped peaks.
    """
    # Compute the original KDE and identify the peaks
    kde_values = kde_func(data, x_range, bandwidth)
    original_peaks, _ = find_peaks(kde_values)
    original_peak_positions = x_range[original_peaks]

    # Initialize a list to store the closest peaks for each bootstrap iteration
    bootstrapped_peaks = {peak: [] for peak in original_peak_positions}

    # Perform bootstrap iterations
    for _ in range(n_bootstrap):
        # Resample data with replacement
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        
        # Compute KDE for the resampled data
        kde_values_resampled = kde_func(resampled_data, x_range, bandwidth)
        
        # Find peaks in the resampled KDE
        resampled_peaks, _ = find_peaks(kde_values_resampled)
        resampled_peak_positions = x_range[resampled_peaks]
        
        # For each original peak, find the closest peak in the resampled data
        for peak in original_peak_positions:
            if len(resampled_peak_positions) > 0:
                closest_peak = min(resampled_peak_positions, key=lambda x: abs(x - peak))
                bootstrapped_peaks[peak].append(closest_peak)
    
    # Compute the mean and standard deviation of the bootstrapped peaks
    peak_means = []
    peak_stds = []
    for peak, resampled_peaks in bootstrapped_peaks.items():
        peak_means.append(np.mean(resampled_peaks))
        peak_stds.append(np.std(resampled_peaks))

    return np.array(peak_means), np.array(peak_stds)

# Function to automatically determine the optimal bandwidth using GridSearchCV
def find_optimal_bandwidth(data):
    """
    Identify the optimal bandwidth for KDE using GridSearchCV.

    Parameters:
        data (array-like): Dataset for which to find the optimal bandwidth.

    Returns:
        optimal_bandwidth (float): Best bandwidth value.
    """
    # Create a grid of bandwidths to test
    bandwidths = np.linspace(0.01, 0.1, 15)  # Adjust the range and step if needed
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
    grid.fit(data[:, None])
    
    return grid.best_params_['bandwidth']

# Function to find the best bin width for the histogram
def find_best_bin_width(data):
    """
    Calculate the bin width for histograms using various methods.

    Parameters:
        data (array-like): Dataset for which to calculate bin width.
        method (str): Method to calculate bin width ('sqrt', 'sturges', 'rice', 'fd').

    Returns:
        bin_width (float): Calculated bin width.
    """
    # Define different binning strategies
    bin_methods = ['fd', 'sturges', 'sqrt', 'doane', 'scott', 'rice']
    best_method = None
    best_peak_count = 0
    best_hist = None
    
    # Test each binning method
    for method in bin_methods:
        hist, bin_edges = np.histogram(data, bins=method, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        peak_count = len(find_peaks(hist)[0])
        
        # Choose the method that results in the most peaks
        if peak_count > best_peak_count:
            best_peak_count = peak_count
            best_method = method
            best_hist = (hist, bin_centers)
    
    return best_method, best_hist

# Load your Pb loss age data
csv_file = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/Monte Carlo Comparison/Mt_Isa_GroupI_v3_200MC_output_1-1400.csv'
df = pd.read_csv(csv_file)

# Filter the Pb loss ages > 100 Ma
df_filtered = df[df['Pb loss age (Ma)'] > 100]
ages = df_filtered['Pb loss age (Ma)'].values

# Define the range for x (age) over which KDE is computed
x_range = np.linspace(100, 1000, 1000)

# Step 1: Find optimal bandwidth
optimal_bandwidth = find_optimal_bandwidth(ages)
print(f"Optimal bandwidth: {optimal_bandwidth}")

# Step 2: Calculate initial KDE with the optimal bandwidth and find peaks
kde_values = calculate_kde(ages, x_range, optimal_bandwidth)
initial_peaks, _ = find_kde_peaks(kde_values, x_range)

# Step 3: Bootstrap peak uncertainties
peak_means, peak_stds = bootstrap_peak_uncertainty(ages, calculate_kde, x_range, optimal_bandwidth, n_bootstrap=1000)

# Step 4: Find the best bin width for the histogram
best_bin_method, best_hist = find_best_bin_width(ages)
print(f"Best bin method: {best_bin_method}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_range, kde_values, color='magenta', label=f'KDE (bw = {optimal_bandwidth:.2f})', linewidth=2)
plt.hist(ages, bins='fd', density=True, alpha=0.7, color='lightgray', label='Pb loss ages', edgecolor='grey')

# Mark the peaks and their uncertainties
for i, peak in enumerate(peak_means):
    plt.errorbar(peak, kde_values[np.argmin(np.abs(x_range - peak))], 
                 xerr=peak_stds[i], fmt='o', color='purple', markersize=5, label=f'Peak: {peak:.1f} Ma ± {peak_stds[i]:.1f} Ma')

plt.xlabel('Pb loss age (Ma)')
plt.ylabel('Density')
plt.xlim(100, 800)
plt.title('Pb Loss Age Distribution with Bootstrapped Peak Uncertainties')
plt.legend()
plt.grid(False)
plt.show()

# Number of unique samples and total data entries before filtering
num_samples_before = df['SampleID'].nunique()
num_spots_before = len(df)

# Number of unique samples and total data entries after filtering Pb loss ages > 100 Ma
df_filtered = df[df['Pb loss age (Ma)'] > 100]
num_samples_after = df_filtered['SampleID'].nunique()
num_spots_after = len(df_filtered)

print(f"Number of unique samples before filtering: {num_samples_before}")
print(f"Total number of data entries before filtering: {num_spots_before}")
print(f"Number of unique samples after filtering: {num_samples_after}")
print(f"Total number of data entries after filtering: {num_spots_after}")