import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import hashlib
from PIL import Image
from collections import defaultdict
from IPython.display import display, Markdown
import random
import math
import io
import sys
import shutil
from tqdm.auto import tqdm

import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
import warnings

warnings.filterwarnings('ignore')


# Imports
import os
import glob
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_sample_images(sample_images_by_class):
    """Create a grid plot of sample images from each class"""
    num_classes = len(sample_images_by_class)
    
    if num_classes == 0:
        return plt.figure()
    
    # Determine grid size based on number of classes
    cols = min(5, num_classes)  # Maximum 3 columns
    rows = math.ceil(num_classes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    
    # Flatten axes for easy indexing if there are multiple rows
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert to list if only one subplot
    
    # Plot each sample image
    for i, (class_name, img_path) in enumerate(sample_images_by_class.items()):
        if i < len(axes):  # Make sure we don't exceed the number of subplots
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{class_name}\n{img.size[0]}x{img.size[1]}")
                axes[i].axis('off')
                img.close()  # Make sure to close the image
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{class_name}\n{str(e)}", 
                             ha='center', va='center')
                axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(sample_images_by_class), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure as an image file
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    
    return fig

def display_dataset_report(report):
    """Display analysis results in Jupyter-friendly format and save to a file"""
    
    # Redirect stdout to capture output for the text file
    original_stdout = sys.stdout
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    
    # Function to print to both notebook and output file
    def print_md(markdown_text):
        # For display, use the original markdown text with emojis
        display(Markdown(markdown_text))
        
        # For the text file, replace emojis with text versions
        plain_text = (
            markdown_text
            .replace("## 📊", "## ")
            .replace("## 🔍", "## ")
            .replace("## 📁", "## ")
            .replace("## 🖼️", "## ")
            .replace("## 📄", "## ")
            .replace("## ❌", "## ")
            .replace("## 📸", "## ")
            .replace("## 🚀", "## ")
            .replace("## 📂", "## ")
            .replace("⚠️", "WARNING:")
            .replace("✅", "OK:")
        )
        print(plain_text.replace("##", "").strip())
    
    # Dataset Folder Structure (for chatbot readability)
    print_md("## 📂 Dataset Folder Structure")
    folder_structure = report['folder_structure']
    root_path = folder_structure['root']
    
    # Display folder structure in a chatbot-friendly format
    structure_text = f"DATASET_PATH: {root_path}\n"
    structure_text += "FOLDER_STRUCTURE:\n"
    for class_info in folder_structure['classes']:
        structure_text += f"  - {class_info['name']} ({class_info['file_count']} images)\n"
    
    display(Markdown(f"```\n{structure_text}\n```"))
    
    # For the text file - explicitly chatbot-friendly format
    print("\nDATASET STRUCTURE FOR CHATBOT REFERENCE:")
    print(structure_text)
    
    # Summary Statistics
    print_md("## 📊 Dataset Summary Statistics")
    summary_data = {
        'Total Classes': [len(report['class_names'])],
        'Total Images': [report['total_images']],
        'Avg Images/Class': [report['total_images']/max(1, len(report['class_names']))],
        'Unique Formats': [len(report['unique_formats'])],
        'Corrupted Files': [len(report['corrupted_files'])]
    }
    summary_df = pd.DataFrame(summary_data)
    display(summary_df.style.set_caption("Key Statistics"))
    print("\nKEY STATISTICS:")
    print(summary_df.to_string(index=False))
    
    # Small Images Analysis
    print_md("## 🔍 Small Image Analysis")
    small_img_data = []
    for threshold, count in report['small_image_counts'].items():
        percent = (count / max(1, report['total_images'])) * 100
        small_img_data.append({
            'Threshold': threshold.replace('below_', '< ') + 'px',
            'Count': count,
            '% of Total': f"{percent:.1f}%"
        })
    small_img_df = pd.DataFrame(small_img_data)
    display(small_img_df.style.set_caption("Small Image Distribution"))
    print("\nSMALL IMAGE DISTRIBUTION:")
    print(small_img_df.to_string(index=False))
    
    # Class Distribution
    print_md("## 📁 Class Distribution")
    class_df = pd.DataFrame(list(report['class_counts'].items()), 
                           columns=['Class', 'Count'])
    if not class_df.empty:
        class_df['% Total'] = (class_df['Count'] / max(1, report['total_images']) * 100).round(1)
        class_df = class_df.sort_values('Count', ascending=False)
        display(class_df.style.bar(subset=['Count'], color='#5fba7d')
               .set_caption(f"Class Distribution (Sorted by Count)"))
        print("\nCLASS DISTRIBUTION (SORTED BY COUNT):")
        print(class_df.to_string(index=False))

    # Image Characteristics
    print_md("## 🖼️ Image Characteristics")
    
    # Resolution Analysis
    if report['resolution_stats']:
        rs = report['resolution_stats']
        res_df = pd.DataFrame({
            'Metric': ['Average', 'Minimum', 'Maximum'],
            'Width': [round(rs['avg_width'], 1), rs['min_width'], rs['max_width']],
            'Height': [round(rs['avg_height'], 1), rs['min_height'], rs['max_height']]
        }).set_index('Metric')
        display(res_df.style.set_caption("Resolution Statistics (Pixels)"))
        print("\nRESOLUTION STATISTICS (PIXELS):")
        print(res_df.to_string())
    
    # Aspect Ratio Analysis
    if report['aspect_ratio_stats']:
        ars = report['aspect_ratio_stats']
        ar_df = pd.DataFrame({
            'Metric': ['Average', 'Median', 'Minimum', 'Maximum', 'Std Dev'],
            'Value': [
                round(ars['avg_ratio'], 2),
                round(ars['median_ratio'], 2),
                round(ars['min_ratio'], 2),
                round(ars['max_ratio'], 2),
                round(ars['std_ratio'], 2)
            ]
        }).set_index('Metric')
        display(ar_df.style.set_caption("Aspect Ratio Statistics (Width/Height)"))
        print("\nASPECT RATIO STATISTICS (WIDTH/HEIGHT):")
        print(ar_df.to_string())
        
        # Aspect Ratio Categories
        if report['aspect_ratio_categories']:
            arc = report['aspect_ratio_categories']
            total = sum(arc.values())
            arc_df = pd.DataFrame({
                'Category': ['Square (0.9-1.1)', 'Portrait (<0.9)', 'Landscape (>1.1)'],
                'Count': [arc['square'], arc['portrait'], arc['landscape']],
                '% of Total': [
                    f"{arc['square']/total*100:.1f}%",
                    f"{arc['portrait']/total*100:.1f}%",
                    f"{arc['landscape']/total*100:.1f}%"
                ]
            })
            display(arc_df.style.set_caption("Aspect Ratio Categories"))
            print("\nASPECT RATIO CATEGORIES:")
            print(arc_df.to_string(index=False))

    # Channel Distribution
    channel_df = pd.DataFrame(list(report['channel_distribution'].items()),
                            columns=['Channels', 'Count'])
    if not channel_df.empty:
        channel_df['Channel Type'] = channel_df['Channels'].map({
            1: 'Grayscale',
            3: 'RGB',
            4: 'RGBA'
        })
        display(channel_df[['Channel Type', 'Count']].style.set_caption("Color Channels"))
        print("\nCOLOR CHANNELS:")
        print(channel_df[['Channel Type', 'Count']].to_string(index=False))

    # File Formats
    print_md("## 📄 File Formats")
    format_df = pd.DataFrame(report['unique_formats'], columns=['Extensions'])
    display(format_df.style.set_caption("Found File Extensions"))
    print("\nFOUND FILE EXTENSIONS:")
    print(format_df.to_string(index=False))

    # Corrupted Files
    print_md("## ❌ Corrupted Files")
    if report['corrupted_files']:
        corrupt_df = pd.DataFrame(report['corrupted_files'], columns=['Path', 'Error'])
        display(Markdown(f"**Total Corrupted:** {len(corrupt_df)}"))
        display(corrupt_df.head(5).style.set_caption("Sample Corrupted Files"))
        print(f"\nTOTAL CORRUPTED: {len(corrupt_df)}")
        print("SAMPLE CORRUPTED FILES:")
        print(corrupt_df.head(5).to_string(index=False))
    else:
        display(Markdown("✅ No corrupted files found"))
        print("\nOK: No corrupted files found")

    # Plot sample images - just once now
    print_md("## 📸 Sample Images")
    fig = plot_sample_images(report['sample_images_by_class'])
    print("\nSample images saved to 'sample_images.png'")
    
    # Display the image
    #img = Image.open('sample_images.png')
    #display(img)
    #img.close()  # Make sure to close this image too
    
    # Reset stdout
    sys.stdout = original_stdout
    
    # Save output to file with explicit UTF-8 encoding
    try:
        with open('dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(output_buffer.getvalue())
        print("Report saved to dataset_analysis_report.txt")
    except UnicodeEncodeError:
        # If UTF-8 encoding fails, try writing with ASCII and replacing problematic characters
        with open('dataset_analysis_report.txt', 'w', encoding='ascii', errors='replace') as f:
            f.write(output_buffer.getvalue())
        print("Report saved to dataset_analysis_report.txt (with some characters replaced)")

def compute_image_hash(img):
    """Compute a hash from image data to detect duplicates"""
    # Convert to a common format and size for comparison
    img_copy = img.copy()
    img_copy = img_copy.resize((64, 64))  # Resize to small dimensions for faster hashing
    img_copy = img_copy.convert("RGB")  # Convert to common format
    
    # Get binary data and compute hash
    data = img_copy.tobytes()
    return hashlib.md5(data).hexdigest()

def analyze_dataset(dataset_path, analyze_only=True):
    """
    Analyze an image dataset without modifying it
    
    Args:
        dataset_path (str): Path to the dataset directory
        analyze_only (bool): If True, only analyze without flagging for deletion
    """
    # Validate dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Initialize data collectors
    classes = []
    class_counts = defaultdict(int)
    formats = set()
    resolutions = []
    aspect_ratios = []
    channels = []
    corrupted_files = []
    image_hashes = {}  # For duplicate detection
    grayscale_images = []
    duplicate_images = []
    rgba_images = []
    small_images = []
    small_image_counts = {
        "below_224": 0,
        "below_128": 0,
        "below_64": 0,
        "below_32": 0
    }
    sample_images_by_class = {}  # Store sample images for display
    folder_structure = {"root": dataset_path, "classes": []}

    # Get list of classes (directory names)
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]

    # Process each class
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Add class to folder structure
        class_info = {"name": class_name, "path": class_dir, "file_count": 0}
        folder_structure["classes"].append(class_info)

        class_images = []
        # Count images in class
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            if not os.path.isfile(file_path):
                continue

            # Collect file formats
            ext = os.path.splitext(filename)[1].lower()
            if ext:
                formats.add(ext)

            # Try to process image file
            img = None
            try:
                img = Image.open(file_path)
                width, height = img.size
                
                # Check image size thresholds for statistics
                if width < 224 or height < 224:
                    small_image_counts["below_224"] += 1
                if width < 128 or height < 128:
                    small_image_counts["below_128"] += 1
                if width < 64 or height < 64:
                    small_image_counts["below_64"] += 1
                if width < 32 or height < 32:
                    small_image_counts["below_32"] += 1
                
                # Calculate aspect ratio
                aspect_ratio = width / height
                aspect_ratios.append(aspect_ratio)
                
                # Add resolution
                resolutions.append((width, height))
                
                # Get channel information
                num_channels = len(img.getbands())
                channels.append(num_channels)
                
                # Determine image characteristics
                is_grayscale = (num_channels == 1)
                is_rgba = (num_channels == 4)
                is_small = (width < 128 or height < 128)
                
                # Track images by characteristics (for potential cleaning)
                if is_grayscale:
                    grayscale_images.append(file_path)
                if is_rgba:
                    rgba_images.append(file_path)
                if is_small:
                    small_images.append(file_path)
                    
                # Compute hash for duplicate detection
                img_hash = compute_image_hash(img)
                if img_hash in image_hashes:
                    duplicate_images.append((file_path, image_hashes[img_hash]))
                else:
                    image_hashes[img_hash] = file_path
                
                # Close the image
                img.close()
                img = None
                
                # Count the image
                class_counts[class_name] += 1
                class_images.append(file_path)
                class_info["file_count"] += 1
                
            except Exception as e:
                corrupted_files.append((file_path, str(e)))
            finally:
                # Ensure image is closed even if an exception occurred
                if img:
                    img.close()

        # Store a random sample image for this class if available
        if class_images:
            sample_images_by_class[class_name] = random.choice(class_images)

    total_images = sum(class_counts.values())

    # Calculate resolution statistics
    res_stats = None
    if resolutions:
        widths, heights = zip(*resolutions)
        res_stats = {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }

    # Calculate aspect ratio statistics
    aspect_ratio_stats = None
    if aspect_ratios:
        aspect_ratio_stats = {
            'avg_ratio': sum(aspect_ratios) / len(aspect_ratios),
            'median_ratio': sorted(aspect_ratios)[len(aspect_ratios) // 2],
            'min_ratio': min(aspect_ratios),
            'max_ratio': max(aspect_ratios),
            'std_ratio': np.std(aspect_ratios)
        }
        
        # Categorize aspect ratios
        aspect_ratio_categories = {
            'square': 0,  # Approximately square (0.9-1.1)
            'portrait': 0,  # Portrait orientation (ratio < 0.9)
            'landscape': 0  # Landscape orientation (ratio > 1.1)
        }
        
        for ratio in aspect_ratios:
            if 0.9 <= ratio <= 1.1:
                aspect_ratio_categories['square'] += 1
            elif ratio < 0.9:
                aspect_ratio_categories['portrait'] += 1
            else:
                aspect_ratio_categories['landscape'] += 1

    # Calculate channels distribution
    channel_dist = defaultdict(int)
    for c in channels:
        channel_dist[c] += 1

    return {
        'class_names': classes,
        'class_counts': dict(class_counts),
        'total_images': total_images,
        'unique_formats': sorted(formats),
        'resolutions': resolutions,
        'resolution_stats': res_stats,
        'aspect_ratio_stats': aspect_ratio_stats,
        'aspect_ratio_categories': aspect_ratio_categories if aspect_ratios else None,
        'aspect_ratios': aspect_ratios,
        'channel_distribution': dict(channel_dist),
        'corrupted_files': corrupted_files,
        'grayscale_images': grayscale_images,
        'duplicate_images': duplicate_images,
        'rgba_images': rgba_images,
        'small_images': small_images,
        'small_image_counts': small_image_counts,
        'sample_images_by_class': sample_images_by_class,
        'folder_structure': folder_structure
    }

def clean_image_dataset(src_path, dest_path, keep_grayscale=False, keep_rgba=False, keep_duplicates=False, min_resolution=None, use_analysis=None):
    """
    Clean image dataset by copying only the desired images to a new location
    
    Args:
        src_path (str): Source dataset path
        dest_path (str): Destination path for cleaned dataset
        keep_grayscale (bool): If False, exclude grayscale images
        keep_rgba (bool): If False, exclude RGBA images
        keep_duplicates (bool): If False, exclude duplicate images 
        min_resolution (int): If provided, exclude images smaller than this resolution
        use_analysis (dict): Optional pre-computed analysis to speed up processing
    """
    # Create base output directory if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    
    # Run analysis if not provided
    if use_analysis is None:
        analysis = analyze_dataset(src_path)
    else:
        analysis = use_analysis
        
    # Create a set of files to exclude
    exclude_files = set()
    
    # Add files to exclude based on parameters
    if not keep_grayscale:
        exclude_files.update(analysis['grayscale_images'])
        
    if not keep_rgba:
        exclude_files.update(analysis['rgba_images'])
        
    if min_resolution is not None:
        exclude_files.update(analysis['small_images'])
        
    if not keep_duplicates:
        # For duplicates, we need to extract the file paths
        for dup_file, _ in analysis['duplicate_images']:
            exclude_files.add(dup_file)
            
    # Track statistics
    stats = {
        'total_processed': 0,
        'total_copied': 0,
        'excluded': {
            'grayscale': 0,
            'rgba': 0,
            'small': 0,
            'duplicate': 0
        },
        'class_counts': defaultdict(int)
    }
    
    # Process each class directory
    for class_name in analysis['class_names']:
        src_class_dir = os.path.join(src_path, class_name)
        if not os.path.isdir(src_class_dir):
            continue
            
        # Create class directory in destination
        dest_class_dir = os.path.join(dest_path, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        # Process files in this class
        for filename in os.listdir(src_class_dir):
            src_file_path = os.path.join(src_class_dir, filename)
            if not os.path.isfile(src_file_path):
                continue
                
            stats['total_processed'] += 1
            
            # Check if file should be excluded
            if src_file_path in exclude_files:
                # Count excluded files by type
                if src_file_path in analysis['grayscale_images']:
                    stats['excluded']['grayscale'] += 1
                elif src_file_path in analysis['rgba_images']:
                    stats['excluded']['rgba'] += 1
                elif src_file_path in analysis['small_images']:
                    stats['excluded']['small'] += 1
                else:
                    # Must be a duplicate
                    stats['excluded']['duplicate'] += 1
                continue
                
            # Copy file to destination
            dest_file_path = os.path.join(dest_class_dir, filename)
            try:
                shutil.copy2(src_file_path, dest_file_path)  # copy2 preserves metadata
                stats['total_copied'] += 1
                stats['class_counts'][class_name] += 1
            except Exception as e:
                print(f"Failed to copy {src_file_path}: {e}")
    
    # Generate and save report
    generate_cleaning_report(src_path, dest_path, stats)
    
    return stats

def generate_cleaning_report(src_path, dest_path, stats):
    """Generate a report about the dataset cleaning process"""
    # Create report buffer
    report = io.StringIO()
    
    # Write header information
    report.write("DATASET CLEANING REPORT\n")
    report.write("======================\n\n")
    report.write(f"Source Dataset: {src_path}\n")
    report.write(f"Cleaned Dataset: {dest_path}\n\n")
    
    # Write summary statistics
    report.write("SUMMARY STATISTICS\n")
    report.write("------------------\n")
    report.write(f"Total files processed: {stats['total_processed']}\n")
    report.write(f"Total files copied: {stats['total_copied']}\n")
    report.write(f"Total files excluded: {stats['total_processed'] - stats['total_copied']}\n\n")
    
    # Write exclusion details
    report.write("EXCLUSION DETAILS\n")
    report.write("----------------\n")
    report.write(f"Grayscale images excluded: {stats['excluded']['grayscale']}\n")
    report.write(f"RGBA images excluded: {stats['excluded']['rgba']}\n")
    report.write(f"Small images excluded: {stats['excluded']['small']}\n")
    report.write(f"Duplicate images excluded: {stats['excluded']['duplicate']}\n\n")
    
    # Write class distribution
    report.write("CLASS DISTRIBUTION IN CLEANED DATASET\n")
    report.write("------------------------------------\n")
    for class_name, count in sorted(stats['class_counts'].items()):
        report.write(f"{class_name}: {count} images\n")
    
    # Print report to notebook
    print("=== Dataset Cleaning Complete ===\n")
    print(f"Source: {src_path}")
    print(f"Destination: {dest_path}\n")
    print(f"Total files processed: {stats['total_processed']}")
    print(f"Total files copied: {stats['total_copied']}")
    print(f"Total files excluded: {stats['total_processed'] - stats['total_copied']}\n")
    
    # Save report to file
    try:
        with open('dataset_cleaning_report.txt', 'w', encoding='utf-8') as f:
            f.write(report.getvalue())
        print("Detailed report saved to dataset_cleaning_report.txt")
    except UnicodeEncodeError:
        with open('dataset_cleaning_report.txt', 'w', encoding='ascii', errors='replace') as f:
            f.write(report.getvalue())
        print("Detailed report saved to dataset_cleaning_report.txt (with some characters replaced)")








##########################################################################################################################







# Functions for outlier analysis

def initialize_model(device):
    """Initialize ResNet50 model with pretrained weights"""
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.to(device).eval(), weights.transforms()

def process_image_directory(root_dir, device, transform, batch_size=32):
    """Process directory and extract features"""
    model, _ = initialize_model(device)
    features, labels, paths = [], [], []
    
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img_tensor).cpu().squeeze().numpy()
                features.append(feat)
                labels.append(class_name)
                paths.append(img_path)
            except Exception as e:
                print(f"Skipped {img_path}: {str(e)}")
    
    return np.array(features), np.array(labels), np.array(paths)

def create_embeddings(features, labels, pca_components=50, umap_params=None):
    """Create supervised UMAP embeddings"""
    umap_params = umap_params or {
        'n_components': 2,
        'target_metric': 'categorical',
        'target_weight': 0.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    le = LabelEncoder()
    y_numeric = le.fit_transform(labels)
    
    # PCA first
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features)
    
    # UMAP
    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(features_pca, y=y_numeric)
    
    return embedding, le, pca, reducer

def detect_outliers(embedding, labels, class_n_neighbors=30, class_contamination=0.05, 
                    global_n_neighbors=75, global_contamination=0.03):
    """Detect outliers using Local Outlier Factor"""
    le = LabelEncoder()
    y_numeric = le.fit_transform(labels)
    
    # Class-wise outliers
    class_outliers = np.zeros(len(labels), dtype=bool)
    for class_id in np.unique(y_numeric):
        mask = y_numeric == class_id
        lof = LocalOutlierFactor(n_neighbors=class_n_neighbors, 
                               contamination=class_contamination)
        class_outliers[mask] = lof.fit_predict(embedding[mask]) == -1
    
    # Global outliers
    global_lof = LocalOutlierFactor(n_neighbors=global_n_neighbors,
                                  contamination=global_contamination)
    global_outliers = global_lof.fit_predict(embedding) == -1
    
    return class_outliers, global_outliers

def create_results_dataframe(embedding, labels, paths, class_outliers, global_outliers):
    """Create comprehensive results dataframe"""
    le = LabelEncoder()
    y_numeric = le.fit_transform(labels)
    
    return pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels,
        'label_encoded': y_numeric,
        'path': paths,
        'is_class_outlier': class_outliers,
        'is_global_outlier': global_outliers
    })

# cell 3: Visualization functions
def plot_umap(df, figsize=(12, 10), alpha=0.6):
    """Plot UMAP embedding with class coloring"""
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        df.x, df.y, 
        c=df.label_encoded, 
        alpha=alpha,
        cmap='tab20'
    )
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=df.label.unique().tolist(),
        title="Classes"
    )
    plt.title("Supervised UMAP Projection")
    plt.show()

def plot_outliers(df, figsize=(14, 10)):
    """Plot outliers on top of UMAP embedding"""
    plt.figure(figsize=figsize)
    
    # Plot normal points
    normal = df[~df.is_class_outlier & ~df.is_global_outlier]
    plt.scatter(normal.x, normal.y, c=normal.label_encoded, 
                alpha=0.3, cmap='tab20', label='Normal')
    
    # Plot class outliers
    class_outliers = df[df.is_class_outlier]
    plt.scatter(class_outliers.x, class_outliers.y, 
                c='red', s=50, label='Class Outliers')
    
    # Plot global outliers
    global_outliers = df[df.is_global_outlier]
    plt.scatter(global_outliers.x, global_outliers.y,
                c='black', marker='x', s=100, label='Global Outliers')
    
    plt.legend()
    plt.title("Outlier Detection Results")
    plt.show()

def display_outlier_stats(df):
    """Show statistics about outliers per class"""
    stats = df.groupby('label').agg(
        Total=('label', 'count'),
        Class_Outliers=('is_class_outlier', 'sum'),
        Global_Outliers=('is_global_outlier', 'sum')
    ).sort_values('Class_Outliers', ascending=False)
    
    stats['Class_%'] = (stats.Class_Outliers / stats.Total * 100).round(1)
    stats['Global_%'] = (stats.Global_Outliers / stats.Total * 100).round(1)
    
    return stats.style.background_gradient(
        cmap='YlOrRd', subset=['Class_%', 'Global_%']
    )

def display_outlier_samples(df, outlier_type='class', n_samples=8, figsize=(16, 8), random_state=42):
    """Display sample outlier images"""
    if outlier_type == 'class':
        subset = df[df.is_class_outlier]
    else:
        subset = df[df.is_global_outlier]
        
    samples = subset.sample(min(n_samples, len(subset)), random_state=random_state)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = axes.ravel()
    
    for ax in axes:
        ax.axis('off')
        
    for idx, (_, row) in enumerate(samples.iterrows()):
        try:
            img = Image.open(row.path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{row.label}\n({row.x:.1f}, {row.y:.1f})", fontsize=8)
        except:
            axes[idx].set_title("Failed to load", fontsize=8)
    
    plt.tight_layout()
    plt.show()

def create_clean_dataset(df, clean_data_root):
    """
    Create a clean dataset by copying non-outlier images to a new directory structure.
    
    Args:
        df (pd.DataFrame): DataFrame containing image paths, labels, and outlier flags
        clean_data_root (str): Path where clean dataset will be created
    """
    # Filter non-outliers
    clean_df = df[(~df['is_class_outlier']) & (~df['is_global_outlier'])]
    
    # Create destination directory structure
    os.makedirs(clean_data_root, exist_ok=True)
    for class_name in clean_df['label'].unique():
        os.makedirs(os.path.join(clean_data_root, class_name), exist_ok=True)
    
    # Copy files with progress bar
    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Copying clean images"):
        src_path = row['path']  # Full source path
        class_name = row['label']
        filename = os.path.basename(src_path)
        dest_path = os.path.join(clean_data_root, class_name, filename)
        
        try:
            shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
        except Exception as e:
            print(f"Failed to copy {src_path}: {e}")
    
    # Print summary
    print_summary(df, clean_df, clean_data_root)

def print_summary(original_df, clean_df, clean_data_root):
    """Print statistics about the dataset cleaning process."""
    print(f"\nDone! Clean dataset created at {clean_data_root}")
    print(f"Original images: {len(original_df)}")
    print(f"Clean images: {len(clean_df)}")
    print(f"Outliers removed: {len(original_df) - len(clean_df)}")
    
    # Count files per class in new directory
    print("\nClass distribution in clean dataset:")
    for class_name in sorted(os.listdir(clean_data_root)):
        class_dir = os.path.join(clean_data_root, class_name)
        if os.path.isdir(class_dir):
            print(f"{class_name}: {len(os.listdir(class_dir))} images")






#####################################################################################################################





# Functions for WebDataset curation


def resize_and_crop_image(img, target_size=224):
    """Resize and center crop image to target_size x target_size."""
    # Convert RGBA images to RGB
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Calculate the scale to resize the smaller dimension to target_size
    width, height = img.size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
        
    # Resize while maintaining aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop to target_size x target_size
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    img = img.crop((left, top, right, bottom))
    return img

def get_dataset_info(input_dir):
    """Get dataset info including class names and counts."""
    classes = []
    class_counts = {}
    class_files = {}
    
    for class_dir in sorted(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            classes.append(class_dir)
            # Get all image files in the class directory
            files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                files.extend(glob.glob(os.path.join(class_path, f'*{ext}')))
            
            class_counts[class_dir] = len(files)
            class_files[class_dir] = files
    
    print(f"Found {len(classes)} classes with {sum(class_counts.values())} total images")
    
    return classes, class_counts, class_files

def create_balanced_test_set(class_files, test_size=0.2):
    """Create a balanced test set where each class has the same number of samples."""
    # Find minimum class count for balanced test set
    min_count = min(len(files) for files in class_files.values())
    min_test_count = int(min_count * test_size)
    
    train_files = []
    test_files = []
    
    for class_name, files in class_files.items():
        # Shuffle files to ensure randomness
        random.shuffle(files)
        # Select the same number of test samples from each class
        test_files_for_class = files[:min_test_count]
        train_files_for_class = files[min_test_count:]
        
        # Add class labels
        test_files.extend([(file, class_name) for file in test_files_for_class])
        train_files.extend([(file, class_name) for file in train_files_for_class])
    
    # Shuffle the datasets
    random.shuffle(train_files)
    random.shuffle(test_files)
    
    print(f"Created balanced test set with {len(test_files)} images ({min_test_count} per class)")
    print(f"Training set has {len(train_files)} images")
    
    return train_files, test_files

def write_webdataset(data_files, output_path, prefix, samples_per_shard=1000):
    """Write data to WebDataset format."""
    os.makedirs(output_path, exist_ok=True)
    
    # Calculate number of shards
    num_shards = max(1, len(data_files) // samples_per_shard)
    
    # Create WebDataset pattern
    pattern = os.path.join(output_path, f"{prefix}-%06d.tar")
    print(f"Writing {len(data_files)} samples to {num_shards} shards at {pattern}")
    
    # Process and write each sample
    current_shard = 0
    current_count = 0
    sink = None
    
    # Use standard tqdm instead of notebook version
    for i, (file_path, class_name) in enumerate(tqdm(data_files, desc=f"Creating {prefix} dataset", ncols=80)):
        # Create a new TarWriter when needed
        if sink is None or current_count >= samples_per_shard:
            if sink is not None:
                sink.close()
            shard_name = pattern % current_shard
            sink = wds.TarWriter(shard_name)
            current_shard += 1
            current_count = 0
        
        try:
            # Load and process image
            img = Image.open(file_path)
            img = resize_and_crop_image(img)
            
            # Create a BytesIO object to save the image
            import io
            byte_array = io.BytesIO()
            img.save(byte_array, format="JPEG")
            
            # Create a sample
            sample_id = f"{class_name}_{i:06d}"
            sample = {
                "__key__": sample_id,
                "jpg": byte_array.getvalue(),
                "cls": class_name,
                "json": {"class": class_name, "id": sample_id}
            }
            
            # Write sample to WebDataset
            sink.write(sample)
            current_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Close the final sink
    if sink is not None:
        sink.close()
    
    print(f"Finished writing {prefix} dataset")

def process_dataset(input_dir, output_dir, test_size=0.2, samples_per_shard=1000):
    """Main function to process dataset with all steps."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset information
    classes, class_counts, class_files = get_dataset_info(input_dir)
    
    # Create balanced train/test split
    train_files, test_files = create_balanced_test_set(class_files, test_size)
    
    # Visualize class distribution
    train_counts = Counter([cls for _, cls in train_files])
    test_counts = Counter([cls for _, cls in test_files])
    
    print("\nClass distribution in train set:")
    for cls in classes:
        print(f"{cls}: {train_counts[cls]}")
        
    print("\nClass distribution in test set:")
    for cls in classes:
        print(f"{cls}: {test_counts[cls]}")
    
    # Create WebDataset files
    write_webdataset(train_files, output_dir, "train", samples_per_shard)
    write_webdataset(test_files, output_dir, "test", samples_per_shard)
    
    # Return summary information
    return {
        "num_classes": len(classes),
        "classes": classes,
        "total_samples": len(train_files) + len(test_files),
        "train_samples": len(train_files),
        "test_samples": len(test_files),
        "train_shards": max(1, len(train_files) // samples_per_shard),
        "test_shards": max(1, len(test_files) // samples_per_shard)
    }

def verify_webdataset(path_pattern, num_samples=5):
    """Load and display a few samples from the WebDataset to verify it works."""
    print(f"Verifying WebDataset at {path_pattern}...")
    
    try:
        # Create dataset with autodecode disabled - this avoids the class name conversion issue
        dataset = wds.WebDataset(path_pattern, shardshuffle=False)
        
        # Convert to a list to ensure we can access items
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            # Manually decode each sample
            try:
                # Convert image bytes to PIL Image
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(sample["jpg"]))
                
                # Decode class name as UTF-8 string
                cls = sample["cls"].decode("utf-8")
                
                samples.append((img, cls))
            except Exception as e:
                print(f"Error processing sample: {e}")
        
        if not samples:
            print(f"No samples found in the dataset. Check if the path pattern '{path_pattern}' is correct.")
            return False
        
        # Display samples
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
        
        # Handle the case of a single sample
        if len(samples) == 1:
            img, cls = samples[0]
            axes.imshow(img)
            axes.set_title(f"Class: {cls}")
            axes.axis("off")
        else:
            for i, (img, cls) in enumerate(samples):
                axes[i].imshow(img)
                axes[i].set_title(f"Class: {cls}")
                axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        print("WebDataset verification successful!")
        return True
    
    except Exception as e:
        print(f"WebDataset verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False



#####################################################################################



# Functions for Animals-10 dataset preparation

def prepare_animal_dataset(dataset_path, destination_dir='./data/raw', verbose=True):
    """
    Prepares the Animals-10 dataset by copying images to folders with English names.
    
    Args:
        dataset_path (str): Path to the downloaded Kaggle dataset
        destination_dir (str): Path where translated folders will be created
        verbose (bool): Whether to print detailed progress information
        
    Returns:
        dict: Summary statistics about the copying process
    """
    
    # Hardcoded Italian to English translations for Animals-10 dataset
    # Fixed to correctly include the spider/ragno translation
    translate = {
        "cane": "dog", 
        "cavallo": "horse", 
        "elefante": "elephant", 
        "farfalla": "butterfly", 
        "gallina": "chicken", 
        "gatto": "cat", 
        "mucca": "cow", 
        "pecora": "sheep", 
        "ragno": "spider",
        "scoiattolo": "squirrel"
    }
    
    # Source directory with original folders
    source_dir = os.path.join(dataset_path, 'raw-img')
    
    # Create destination directory
    os.makedirs(destination_dir, exist_ok=True)
    
    if verbose:
        print(f"Source directory: {source_dir}")
        print(f"Destination directory: {destination_dir}")
        print("\nCopying folders with translated names...")
    
    # Get list of folders in source directory
    if os.path.exists(source_dir):
        folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    else:
        if verbose:
            print(f"Source directory not found: {source_dir}")
        return {"error": "Source directory not found"}
    
    successful_copies = 0
    failed_copies = []
    class_stats = {}
    
    # Process each folder
    for folder in folders:
        # Get the English translation if available
        if folder in translate:
            translated_name = translate[folder]
            
            # Source and destination paths
            src_path = os.path.join(source_dir, folder)
            dst_path = os.path.join(destination_dir, translated_name)
            
            try:
                # Create destination folder if it doesn't exist
                os.makedirs(dst_path, exist_ok=True)
                
                # Get list of files
                files = os.listdir(src_path)
                
                # Copy all files with progress bar if verbose
                if verbose:
                    print(f"Copying {folder} → {translated_name} ({len(files)} files)")
                
                for file in (tqdm(files) if verbose else files):
                    src_file = os.path.join(src_path, file)
                    dst_file = os.path.join(dst_path, file)
                    shutil.copy2(src_file, dst_file)
                
                successful_copies += 1
                class_stats[translated_name] = len(files)
                
            except Exception as e:
                failed_copies.append((folder, str(e)))
                if verbose:
                    print(f"✗ Failed to copy {folder}: {e}")
        else:
            if verbose:
                print(f"⚠ No translation found for '{folder}', skipping")
    
    # Generate summary statistics
    summary = {
        "successful_copies": successful_copies,
        "failed_copies": failed_copies,
        "total_folders": len(folders),
        "class_statistics": class_stats,
        "total_images": sum(class_stats.values()) if class_stats else 0
    }
    
    # Display summary 
    print(f"\nSummary: {successful_copies} folders copied successfully")
    if failed_copies:
        print(f"{len(failed_copies)} folders failed to copy:")
        for folder, error in failed_copies:
            print(f"  - {folder}: {error}")
    else:
        print("No errors occurred")
        
    if class_stats:
        print("\nContents of destination directory:")
        for class_name, count in class_stats.items():
            print(f"  - {class_name}: {count} files")
    
    return summary

