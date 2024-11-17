import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import os
import re

# Function to calculate KDE with dynamic bandwidth
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

# Function to find peaks in KDE and return peak positions and heights
def find_kde_peaks(kde_values, x_range, height=None):  # Set height to None by default
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
    if height:
        peaks, _ = find_peaks(kde_values, height=height)
    else:
        peaks, _ = find_peaks(kde_values)  # Detect all peaks without height threshold
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

# Function to find optimal bandwidth using GridSearchCV with a minimum constraint
def find_optimal_bandwidth(data, min_bandwidth, max_bandwidth):
    """
    Identify the optimal bandwidth for KDE using GridSearchCV.

    Parameters:
        data (array-like): Dataset for which to find the optimal bandwidth.

    Returns:
        optimal_bandwidth (float): Best bandwidth value.
    """
    bandwidths = np.linspace(min_bandwidth, max_bandwidth, 20)
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
    grid.fit(data[:, None])
    
    optimal_bandwidth = grid.best_params_['bandwidth']
    return max(optimal_bandwidth, min_bandwidth)

# Function to calculate bin width for histograms
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

# Function to load CSV files
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

# Function to extract step size from the filename
def extract_step_size(label):
    """
    Extract the numeric step-size from the file label.

    Parameters:
        label (str): Filename label.

    Returns:
        step_size (int): Step-size extracted from the label.
    """
    match = re.search(r'(\d+)Ma', label)  # Extract the step size before 'Ma'
    if match:
        return match.group(1)
    return None

# Function to sort files based on step size
def sort_files_by_step_size(file_paths):
    """
    Sort MC simulation files by their step sizes.

    Parameters:
        file_paths (dict): Dictionary of file labels and their paths.

    Returns:
        sorted_file_paths (dict): Dictionary of sorted file paths.
    """
    # Filter out files that do not have valid step size
    valid_files = {k: v for k, v in file_paths.items() if extract_step_size(k) is not None}
    return dict(sorted(valid_files.items(), key=lambda x: int(extract_step_size(x[0]))))

# Function to compute AUC (area under the curve)
def compute_auc(kde_values, x_range):
    auc = np.trapz(kde_values, x=x_range)
    return auc

# Function to perform pairwise K-S tests for distributions
def perform_ks_test(data1, data2):
    stat, p_val = ks_2samp(data1, data2)
    return stat, p_val

# Function to plot CDF for comparisons
def plot_cdf_simulations(dataframes, labels, colors):
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
        sorted_data = np.sort(filtered_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot CDF
        plt.plot(sorted_data, cumulative, label=label, color=color)
    
    plt.xlabel('Pb loss age (Ma)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Pb-loss Ages for Different Step Sizes (> 100 Ma)')
    plt.legend(title='Step size')
    plt.grid(False)
    plt.show()

# Set the folder path
folder_path = '/Users/lucymathieson/Library/CloudStorage/OneDrive-Curtin/Pb_loss-KIRKLC-SE10523/Pb loss modelling/Mount Isa/Pb loss model/Outputs/Step size comparison'

# Load the files
file_paths = load_mc_files(folder_path)

# Sort the files by step size
sorted_file_paths = sort_files_by_step_size(file_paths)

# Define colors for KDE lines (vibrant colors)
kde_colors = ['yellow', 'magenta', 'cyan', 'green', 'blue', 'cyan', 'darkorchid', 'red']

# Define the x range for KDE
x_range = np.linspace(100, 1000, 1000)

# Initialize figure for plotting KDE, histograms, and peaks
plt.figure(figsize=(10, 6))

# For legend labels (Step size)
legend_labels = []

# Initialize data for CDF comparison
dataframes = []

# Dictionary to store K-S test results
ks_test_results = {}
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

    # Extract step size from the label
    step_size = extract_step_size(label)

    # Append dataframe for CDF
    dataframes.append(df_filtered)
    
    # Adjust the range depending on your needs
    optimal_bandwidth = find_optimal_bandwidth(ages, min_bandwidth=0.1, max_bandwidth=0.2)

    # Step 2: Calculate KDE with the optimal bandwidth
    kde_values = calculate_kde(ages, x_range, optimal_bandwidth)
    peak_positions, _ = find_kde_peaks(kde_values, x_range)

    # Step 3: Bootstrap peak uncertainties
    if len(peak_positions) > 0:
        peak_means, peak_stds = bootstrap_peak_uncertainty(ages, calculate_kde, x_range, optimal_bandwidth, n_bootstrap=1000)

        # Print the peaks with uncertainties (errors)
        print(f"Step size: {step_size} Ma")
        print(f"Peak positions: {peak_means}")
        print(f"Peak errors (standard deviation): {peak_stds}")
        print('---')

        # Mark the peaks with uncertainties on the plot
        for i, peak in enumerate(peak_means):
            plt.errorbar(peak, kde_values[np.argmin(np.abs(x_range - peak))], 
                         xerr=peak_stds[i], fmt='o', color=kde_colors[idx % len(kde_colors)], markersize=5)
    else:
        print(f"No peaks detected for step size: {step_size} Ma")
        print('---')

    # Calculate bin width for histograms
    bin_width = calculate_bin_width(ages, method='fd')
    bins = np.arange(min(ages), max(ages) + bin_width, bin_width)

    # Plot histograms in grey with dynamic bin width
    plt.hist(ages, bins=bins, density=True, alpha=0.7, color='lightgray', edgecolor='grey', label=None)

    # Plot KDE with vibrant colors and step size label
    plt.plot(x_range, kde_values, color=kde_colors[idx % len(kde_colors)],linewidth = 2, label=f"{step_size} Ma")

    # Store step size for the legend
    legend_labels.append(f"{step_size} Ma")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Add labels and customize the legend
plt.xlabel('Pb loss age (Ma)')
plt.ylabel('Density')
plt.title('Pb Loss Age Distribution with Bootstrapped Peak Uncertainties')
plt.xlim(100, 1000)
plt.grid(False)

# Customize legend to display simulation counts under a title
plt.legend(title='Simulation count', labels=legend_labels)

# Create an inset axis for the CDF plot
ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='upper right')

# Plot the CDFs in the inset
for df, label, color in zip(dataframes, legend_labels, kde_colors):
    filtered_data = df[df['Pb loss age (Ma)'] > 100]['Pb loss age (Ma)']
    sorted_data = np.sort(filtered_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax_inset.plot(sorted_data, cumulative, label=label, color=color, linewidth=1)

# Customize the inset plot
ax_inset.set_xlabel('Pb loss age (Ma)', fontsize=7)
ax_inset.set_ylabel('Cumulative Probability', fontsize=7)
ax_inset.set_title('CDF', fontsize=7)
ax_inset.tick_params(axis='both', which='major', labelsize=7)
ax_inset.set_xlim(100, 800)

plt.show()


# Perform pairwise K-S tests and display the results
for i in range(len(dataframes)):
    """
    Perform pairwise K-S tests between multiple datasets.

    Parameters:
        dataframes (list): List of dataframes to compare.
        labels (list): List of labels for each dataframe.

    Prints:
        Pairwise K-S test results (statistic, p-value).
    """
    for j in range(i+1, len(dataframes)):
        stat, p_val = perform_ks_test(dataframes[i]['Pb loss age (Ma)'], dataframes[j]['Pb loss age (Ma)'])
        ks_test_results[f"{legend_labels[i]} vs {legend_labels[j]}"] = (stat, p_val)

# Print K-S test results
print("Pairwise K-S test results (statistic, p-value):")
for comparison, result in ks_test_results.items():
    print(f"{comparison}: {result}")

# Plot CDF comparisons
plot_cdf_simulations(dataframes, legend_labels, kde_colors)
