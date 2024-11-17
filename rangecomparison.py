import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks
from sklearn.utils import resample
import os
import re

# Function to calculate KDE
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

# Function to find peaks in KDE
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

# Function to find optimal bandwidth
def find_optimal_bandwidth(data):
    """
    Identify the optimal bandwidth for KDE using GridSearchCV.

    Parameters:
        data (array-like): Dataset for which to find the optimal bandwidth.

    Returns:
        optimal_bandwidth (float): Best bandwidth value.
    """
    bandwidths = np.linspace(0.01, 0.1, 20)
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
    grid.fit(data[:, None])
    return grid.best_params_['bandwidth']

# Function to calculate bin width using different methods
def calculate_bin_width(data, method='sqrt'):
    """
    Calculate the bin width for histograms using various methods.

    Parameters:
        data (array-like): Dataset for which to calculate bin width.
        method (str): Method to calculate bin width ('sqrt', 'sturges', 'rice', 'fd').

    Returns:
        bin_width (float): Calculated bin width.
    """
    n = len(data)
    if method == 'sqrt':
        bin_width = (max(data) - min(data)) / np.sqrt(n)
    elif method == 'sturges':
        bin_width = (max(data) - min(data)) / (np.log2(n) + 1)
    elif method == 'rice':
        bin_width = (max(data) - min(data)) / (2 * n**(1/3))
    elif method == 'fd':
        bin_width = 2 * np.subtract(*np.percentile(data, [75, 25])) / n**(1/3)
    else:
        raise ValueError("Invalid method. Choose from 'sqrt', 'sturges', 'rice', 'fd'.")
    return bin_width

# Function to load multiple CSV files
def load_mc_files(folder_path):
    """
    Load MC simulation files from a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing MC files.

    Returns:
        file_paths (dict): Dictionary of file labels and their full paths.
    """
    file_paths = {}
    for filename in os.listdir(folder_path):
        if 'MC' in filename and filename.endswith('.csv'):
            label = filename.replace('.csv', '')  # Label by filename without the extension
            full_path = os.path.join(folder_path, filename)
            file_paths[label] = full_path
    return file_paths

# Function to extract the correct age range from the filename and sort them
def extract_numeric_range(label):
    """
    Extract the numeric age range from the file label.

    Parameters:
        label (str): Filename label.

    Returns:
        numeric_range (int): Numeric age range extracted from the label.
    """
    match = re.search(r'1-(\d+)', label)  # Extract the numeric part of the range
    if match:
        return int(match.group(1))
    return None

# Function to sort the file paths based on the numeric range in the filenames
def sort_files_by_range(file_paths):
    """
    Sort MC simulation files by their numeric age ranges.

    Parameters:
        file_paths (dict): Dictionary of file labels and their paths.

    Returns:
        sorted_file_paths (dict): Dictionary of sorted file paths.
    """
    return dict(sorted(file_paths.items(), key=lambda x: extract_numeric_range(x[0])))

# Function to plot CDF for comparisons
def plot_cdf(dataframes, labels, colors):
    """
    Plot cumulative distribution functions (CDFs) for multiple datasets.

    Parameters:
        dataframes (list): List of dataframes to plot.
        labels (list): List of labels for each dataset.
        colors (list): List of colors for each dataset's plot.
    """
    plt.figure(figsize=(10, 6))
    
    for df, label, color in zip(dataframes, labels, colors):
        filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
        if len(filtered_data) == 0:
            print(f"No valid data in CDF for {label}")
            continue

        sorted_data = np.sort(filtered_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot CDF
        plt.plot(sorted_data, cumulative, label=label, color=color)
    
    plt.xlabel('Pb loss age (Ma)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Pb-loss Ages for Different Step Sizes')
    plt.legend()
    plt.grid(False)
    plt.show()

# Function to perform pairwise K-S tests for CDFs
def perform_ks_test_for_cdf(dataframes, labels):
    """
    Perform pairwise K-S tests between multiple datasets.

    Parameters:
        dataframes (list): List of dataframes to compare.
        labels (list): List of labels for each dataframe.

    Prints:
        Pairwise K-S test results (statistic, p-value).
    """
    ks_results = {}
    for i in range(len(dataframes)):
        for j in range(i + 1, len(dataframes)):
            data1 = np.sort(dataframes[i]['Pb loss age (Ma)'])
            data2 = np.sort(dataframes[j]['Pb loss age (Ma)'])
            if len(data1) == 0 or len(data2) == 0:
                print(f"Skipping K-S test for {labels[i]} vs {labels[j]} due to empty data.")
                continue
            stat, p_val = ks_2samp(data1, data2)
            ks_results[f"{labels[i]} vs {labels[j]}"] = (stat, p_val)
    
    print("K-S Test Results for CDFs (Statistic, p-value):")
    for comparison, result in ks_results.items():
        print(f"{comparison}: {result}")

# Set folder path
folder_path = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/8% disc filter/100 MC'

# Load files
file_paths = load_mc_files(folder_path)

# Sort files based on the numeric age range
sorted_file_paths = sort_files_by_range(file_paths)

# Define custom colors for KDE lines (explicit vibrant colors)
kde_colors = ['yellow', 'magenta', 'cyan', 'green', 'blue', 'darkorange', 'darkorchid', 'red']

# Define the range for x (age) over which KDE is computed
x_range = np.linspace(100, 1000, 1000)

# Initialize for storing data for CDF and K-S tests
dataframes = []
labels = []

plt.figure(figsize=(10, 6))

for idx, (label, path) in enumerate(sorted_file_paths.items()):
    print(f"Processing file: {label}")
    df = pd.read_csv(path)
    df['Pb loss age (Ma)'] = pd.to_numeric(df['Pb loss age (Ma)'], errors='coerce')
    df_filtered = df[df['Pb loss age (Ma)'] > 100]
    ages = df_filtered['Pb loss age (Ma)'].values

    if len(ages) == 0:
        print(f"No valid data in file {label}. Skipping.")
        continue

    # Store the dataframe and label for CDF and K-S tests
    dataframes.append(df_filtered)
    labels.append(f"1-{extract_numeric_range(label)} Ma")

    # Step 1: Find optimal bandwidth for each dataset
    optimal_bandwidth = find_optimal_bandwidth(ages)

    # Step 2: Calculate KDE and find peaks
    kde_values = calculate_kde(ages, x_range, optimal_bandwidth)
    initial_peaks, _ = find_kde_peaks(kde_values, x_range)

    # Step 3: Bootstrap peak uncertainties
    peak_means, peak_stds = bootstrap_peak_uncertainty(ages, calculate_kde, x_range, optimal_bandwidth, n_bootstrap=1000)

    # Calculate the bin width for the histogram
    bin_width = calculate_bin_width(ages, method='fd')
    bins = np.arange(min(ages), max(ages) + bin_width, bin_width)

    # Plot histogram in grey with dynamic bin width
    plt.hist(ages, bins=bins, density=True, alpha=0.7, color='lightgray', edgecolor='grey', label=None)

    # Plot KDE with vibrant color based on the file index
    plt.plot(x_range, kde_values, color=kde_colors[idx % len(kde_colors)], label=f"1-{extract_numeric_range(label)} Ma", linewidth = 2)

    # Mark the peaks in black
    for i, peak in enumerate(peak_means):
        plt.errorbar(peak, kde_values[np.argmin(np.abs(x_range - peak))], 
                    xerr=peak_stds[i], fmt='o', color=kde_colors[idx % len(kde_colors)], markersize=5)

# Add labels and legend for KDE and histogram plot
plt.xlabel('Pb loss age (Ma)')
plt.ylabel('Density')
plt.title('Age Range')
plt.legend(loc='upper left')
plt.xlim(100, 1000)
plt.grid(False)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Add CDF inset
ax_inset = inset_axes(plt.gca(), width="37%", height="37%", loc='upper right')
for df, label, color in zip(dataframes, labels, kde_colors):
    filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    sorted_data = np.sort(filtered_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax_inset.plot(sorted_data, cumulative, label=label, color=color, linewidth = 0.7)

ax_inset.set_xlabel('Pb loss age (Ma)', fontsize=7)
ax_inset.set_ylabel('Cumulative Probability', fontsize=7)
ax_inset.set_title('CDF', fontsize=7)
ax_inset.set_xlim(450, 500)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
# ax_inset.legend(fontsize=6)
ax_inset.grid(False)

# Show plot with inset
plt.show()

# Plot the CDFs after KDE and histogram plot
# plot_cdf(dataframes, labels, kde_colors)

# Perform the K-S tests after CDFs
perform_ks_test_for_cdf(dataframes, labels)
