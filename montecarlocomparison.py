import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import os
import re

# Function to calculate KDE
def calculate_kde(data, x_range, bandwidth):
    """
    Calculate the Kernel Density Estimate (KDE) for a dataset.
    
    Parameters:
    - data (ndarray): The input data.
    - x_range (ndarray): The range of x-values for KDE evaluation.
    - bandwidth (float): The bandwidth for KDE calculation.
    
    Returns:
    - ndarray: KDE values corresponding to the x_range.
    """
    kde = gaussian_kde(data, bw_method=bandwidth)
    kde_values = kde(x_range)
    return kde_values

# Function to find peaks in KDE
def find_kde_peaks(kde_values, x_range, height=0.1):
    """
    Identify peaks in the KDE curve.
    
    Parameters:
    - kde_values (ndarray): KDE values.
    - x_range (ndarray): x-values corresponding to the KDE.
    - height (float): Minimum height for peak detection.
    
    Returns:
    - tuple: x-values of detected peaks and their corresponding KDE values.
    """
    peaks, _ = find_peaks(kde_values, height=height)
    peak_positions = x_range[peaks]
    return peak_positions, kde_values[peaks]

# Function to bootstrap peak uncertainties
def bootstrap_peak_uncertainty(data, kde_func, x_range, bandwidth, n_bootstrap=1000):
    """
    Perform bootstrap resampling to calculate uncertainties of detected peaks.
    
    Parameters:
    - data (ndarray): Input dataset.
    - kde_func (function): KDE calculation function.
    - x_range (ndarray): Range of x-values for KDE evaluation.
    - bandwidth (float): Bandwidth for KDE calculation.
    - n_bootstrap (int): Number of bootstrap resampling iterations.
    
    Returns:
    - tuple: Mean and standard deviation of bootstrapped peaks.
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
    Determine the optimal bandwidth for KDE using cross-validation.
    
    Parameters:
    - data (ndarray): Input dataset.
    
    Returns:
    - float: Optimal bandwidth.
    """
    bandwidths = np.linspace(0.01, 0.1, 20)
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
    grid.fit(data[:, None])
    return grid.best_params_['bandwidth']

# Function to calculate bin width using different methods
def calculate_bin_width(data, method='fd'):
    """
    Calculate histogram bin width using a specified method.
    
    Parameters:
    - data (ndarray): Input dataset.
    - method (str): Method for bin width calculation ('fd', 'sturges', 'rice', 'sqrt').
    
    Returns:
    - float: Bin width.
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
    Load CSV files containing Monte Carlo simulation outputs.
    
    Parameters:
    - folder_path (str): Directory path containing the CSV files.
    
    Returns:
    - dict: Mapping of file labels to file paths.
    """
    file_paths = {}
    for filename in os.listdir(folder_path):
        if 'MC' in filename and filename.endswith('.csv'):
            label = filename.replace('.csv', '')  # Label by filename without the extension
            full_path = os.path.join(folder_path, filename)
            file_paths[label] = full_path
    return file_paths

# Function to extract the number of Monte Carlo simulations from the filename
def extract_simulation_count(label):
    """
    Extract the number of Monte Carlo simulations from a filename.
    
    Parameters:
    - label (str): Filename label.
    
    Returns:
    - str: Extracted simulation count.
    """
    match = re.search(r'(\d+)MC', label)  # Extract the numeric part before 'MC'
    if match:
        return match.group(1)
    return None

# Function to sort the file paths based on the simulation count
def sort_files_by_simulation_count(file_paths):
    """
    Sort file paths based on the simulation count extracted from filenames.
    
    Parameters:
    - file_paths (dict): Mapping of file labels to file paths.
    
    Returns:
    - dict: Sorted file paths.
    """
    return dict(sorted(file_paths.items(), key=lambda x: int(extract_simulation_count(x[0]))))

# Function to plot CDF for comparisons
def plot_cdf_simulations(dataframes, labels, colors):
    plt.figure(figsize=(10, 6))
    
    for df, label, color in zip(dataframes, labels, colors):
        filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
        sorted_data = np.sort(filtered_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot CDF
        plt.plot(sorted_data, cumulative, label=label, color=color)

    plt.xlabel('Pb loss age (Ma)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Pb Loss Age for Monte Carlo Simulations')
    plt.legend(title='Simulations')
    plt.grid(True)
    plt.show()

# Function to perform pairwise K-S tests
def perform_ks_tests(dataframes, labels):
    """
    Perform pairwise Kolmogorov-Smirnov tests between simulation outputs.
    
    Parameters:
    - dataframes (list): List of DataFrames containing simulation data.
    - labels (list): Labels for the simulations.
    """
    ks_results = {}
    for i in range(len(dataframes)):
        for j in range(i + 1, len(dataframes)):
            data1 = np.sort(dataframes[i]['Pb loss age (Ma)'])
            data2 = np.sort(dataframes[j]['Pb loss age (Ma)'])
            stat, p_val = ks_2samp(data1, data2)
            ks_results[f"{labels[i]} vs {labels[j]}"] = (stat, p_val)
    
    # Print the results
    print("\nK-S Test Results:")
    for comparison, result in ks_results.items():
        print(f"{comparison}: K-S Statistic = {result[0]}, p-value = {result[1]}")

# Set folder path
folder_path = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/Monte Carlo Comparison'

# Load files
file_paths = load_mc_files(folder_path)

# Sort files based on the numeric age range
sorted_file_paths = sort_files_by_simulation_count(file_paths)

# Define custom colors for KDE lines (explicit vibrant colors)
kde_colors = ['green', 'orange', 'cyan', 'magenta', 'blue', 'darkorange', 'darkorchid', 'yellow']

# Define the range for x (age) over which KDE is computed
x_range = np.linspace(100, 1000, 1000)

# Initialize figure for plotting
plt.figure(figsize=(10, 6))

# For legend labels (Simulation count)
legend_labels = []

# Store data for CDF plotting and K-S testing
dataframes = []

# Loop through each sorted file and process
for idx, (label, path) in enumerate(sorted_file_paths.items()):
    print(f"Processing file: {label}")
    df = pd.read_csv(path)
    df['Pb loss age (Ma)'] = pd.to_numeric(df['Pb loss age (Ma)'], errors='coerce')
    df_filtered = df[df['Pb loss age (Ma)'] > 100]
    ages = df_filtered['Pb loss age (Ma)'].values

    if len(ages) == 0:
        print(f"No valid data in file {label}. Skipping.")
        continue

    # Append dataframe for CDF and K-S tests
    dataframes.append(df_filtered)

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

    # Extract the simulation count from the filename for the legend
    sim_count = extract_simulation_count(label)
    legend_labels.append(sim_count)  # Store the extracted simulation count for the legend

    # Plot KDE with vibrant color based on the file index
    plt.plot(x_range, kde_values, color=kde_colors[idx % len(kde_colors)], label=f"{sim_count}", linewidth = 1.5)

    # Mark the peaks in black
    for i, peak in enumerate(peak_means):
        plt.errorbar(peak, kde_values[np.argmin(np.abs(x_range - peak))], 
                     xerr=peak_stds[i], fmt='o', color=kde_colors[idx % len(kde_colors)], markersize=5)
                     
# Add labels and customize the legend
plt.xlabel('Pb loss age (Ma)')
plt.ylabel('Density')
plt.xlim(100, 1000)

# Customize legend to display simulation counts under a title
plt.legend(title='Simulation count', labels=legend_labels)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.grid(False)

# Customize legend to display simulation counts under a title
plt.legend(title='Simulation count', labels=legend_labels)

# Create an inset axis for the CDF plot
ax_inset = inset_axes(plt.gca(), width="37%", height="37%", loc='upper right')

# Plot the CDFs in the inset
for df, label, color in zip(dataframes, legend_labels, kde_colors):
    filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    sorted_data = np.sort(filtered_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax_inset.plot(sorted_data, cumulative, label=label, color=color, linewidth=0.7)

# Customize the inset plot
ax_inset.set_xlabel('Pb loss age (Ma)', fontsize=7)
ax_inset.set_ylabel('Cumulative Probability', fontsize=7)
ax_inset.set_title('CDF', fontsize=10)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
ax_inset.set_xlim(100, 800)

plt.show()
# Perform pairwise K-S tests and print the results
perform_ks_tests(dataframes, legend_labels)
