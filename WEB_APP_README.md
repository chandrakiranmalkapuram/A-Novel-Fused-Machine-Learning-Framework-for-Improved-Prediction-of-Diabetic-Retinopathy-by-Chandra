# Diabetic Retinopathy Detection Web Application

A modern, interactive web application for detecting and classifying Diabetic Retinopathy (DR) using machine learning. Built with **Streamlit** and powered by GLCM features and ensemble methods.

---

## 🎯 Features

✅ **User-Friendly Interface** - Upload datasets through a modern web UI  
✅ **Interactive Processing** - Real-time model training with progress tracking  
✅ **Multiple ML Models** - SVM, ANN, Fusion Framework, and CNN baseline  
✅ **Rich Visualizations** - Comparison charts, confusion matrices, feature distributions  
✅ **Result Download** - Export metrics, plots, and predictions as CSV/PNG  
✅ **Statistical Analysis** - Wilcoxon signed-rank tests and per-fold performance metrics  
✅ **Responsive Design** - Works seamlessly on desktop and tablet browsers  

---

## 📋 Project Structure

```
dr_project/
├── app.py                  # Main Streamlit web application
├── processing.py           # Core ML processing module
├── requirements.txt        # Python dependencies
├── uploads/               # Temporary directory for uploaded files
├── outputs/               # Directory for processed results
├── data/                  # Original dataset (if using Kaggle data)
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   └── test_images/
├── results/               # Previous results (if running original script)
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /path/to/dr_project

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### 3. Use the Application

**Step 1: Upload Dataset (📤 Upload Dataset)**
- Prepare your dataset:
  - **CSV file** with columns: `id_code`, `diagnosis`
  - **Image files** (PNG/JPG) matching the IDs in your CSV
- Upload through the interface
- Click "Load Dataset for Processing"

**Step 2: Train Models (🔬 Processing)**
- Select which models to train:
  - ✓ SVM (fast, ~5-10 minutes)
  - ✓ ANN (fast, ~5-10 minutes)
  - ✓ Fusion (fast, ~10-15 minutes)
  - ☐ CNN (very slow, requires GPU)
- Click "Start Training"
- Monitor progress in real-time

**Step 3: View Results (📈 Results & Analysis)**
- See summary statistics for all models
- Explore interactive visualizations
- Download results as CSV files
- Export charts as PNG images

---

## 📁 File Descriptions

### `app.py` - Main Web Application
The Streamlit application with three pages:

1. **📤 Upload Dataset Page**
   - Upload CSV with `id_code` and `diagnosis` columns
   - Upload corresponding retinal images (PNG/JPG)
   - Automatic validation of dataset structure
   - Preview of uploaded data

2. **🔬 Processing Page**
   - Select which models to train (SVM, ANN, Fusion, CNN)
   - Real-time progress tracking
   - Feature extraction and model training

3. **📈 Results & Analysis Page**
   - Summary statistics table
   - Four tabs for different visualizations:
     - Model Comparison Chart (bar charts with error bars)
     - Confusion Matrix (Fusion model)
     - GLCM Feature Distributions (by DR grade)
     - Per-Fold Performance (line plots)
   - Download results as CSV
   - Per-model detailed metrics

### `processing.py` - Core Processing Module
Contains all ML and image processing functions:

**Image Processing:**
- `preprocess_image()` - Resize, normalize, apply CLAHE
- `extract_glcm_features()` - Extract texture features
- `extract_features_from_images()` - Batch feature extraction

**Model Training:**
- `run_svm()` - Train SVM with GridSearchCV
- `run_ann()` - Train Neural Network
- `run_fusion()` - Fuzzy fusion of SVM + ANN
- `run_cnn()` - CNN baseline training

**Analysis & Visualization:**
- `compute_summary_stats()` - Generate summary table
- `wilcoxon_test()` - Statistical significance testing
- `plot_comparison_chart()` - Model comparison visualization
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_glcm_distributions()` - Feature distribution plots
- `plot_per_fold()` - Per-fold performance tracking

### `requirements.txt` - Dependencies
All Python packages needed for the application:

**ML Libraries:**
- numpy, pandas, scipy, scikit-learn, scikit-image, tensorflow

**Image Processing:**
- opencv-python, pillow

**Visualization:**
- matplotlib, seaborn, plotly

**Web Framework:**
- streamlit, streamlit-option-menu

---

## 💾 Input Format

### CSV File Format

Your CSV file must contain at least these columns:

```csv
id_code,diagnosis
image_001,0
image_002,1
image_003,2
image_004,3
image_005,4
```

**Column Descriptions:**
- `id_code`: Image filename (without extension). Example: `image_001`
- `diagnosis`: DR severity grade (0-4):
  - 0 = No DR
  - 1 = Mild
  - 2 = Moderate
  - 3 = Severe
  - 4 = Proliferative DR (PDR)

### Image Files

- **Format**: PNG or JPG/JPEG
- **Filename**: Must match the `id_code` in your CSV
  - Example: `image_001.png`, `image_001.jpg`
- **Size**: Any size (will be resized to 224×224)
- **Color**: RGB fundus images (retinal photographs)

---

## 📊 Output Files

When you download results from the web app, you get:

### CSV Results
- `results_summary.csv` - Mean and standard deviation for all metrics across all folds

### Visualizations (PNG)
- `comparison_chart.png` - Bar chart comparing model performance
- `confusion_matrix_fusion.png` - Confusion matrix for Fusion model
- `glcm_distributions.png` - Feature distribution plots by DR grade
- `per_fold_performance.png` - Per-fold metrics across folds

---

## 🔄 Comparison: Original Script vs Web Application

### Original Script (`main.py`)

```
❌ Run only locally in terminal
❌ Manual dataset file management
❌ Long wait times with no UI feedback
❌ Results saved only to local files
❌ No interactive exploration of results
```

### Web Application (`app.py`)

```
✅ Access from any browser
✅ Easy file upload interface
✅ Real-time progress tracking
✅ Results displayed immediately on webpage
✅ Interactive tables and charts
✅ Download results with one click
✅ Beautiful, modern interface
```

---

## 🛠️ Code Migration Guide

### Where Your Original Code Went

**Original `main.py` → New Structure:**

| Original Function | New Location | Notes |
|---|---|---|
| `banner()`, `step()`, `done()` | `app.py` | UI helpers converted to Streamlit messages |
| `setup_kaggle()` | Removed | Users upload their own data |
| `preprocess_image()` | `processing.py` | Unchanged, reused as-is |
| `extract_glcm_features()` | `processing.py` | Unchanged, reused as-is |
| `load_and_extract()` | `processing.py` (renamed) | Now `extract_features_from_images()` |
| `run_svm()` | `processing.py` | Modified to return fold details |
| `run_ann()` | `processing.py` | Modified to return fold details |
| `run_fusion()` | `processing.py` | Modified to return fold details |
| `run_cnn()` | `processing.py` | Adapted for web (images not pre-loaded) |
| `compute_metrics()` | `processing.py` | Unchanged |
| `print_results_table()` | `app.py` | Converted to Streamlit dataframe display |
| `save_results_csv()` | `app.py` | Integrated with Streamlit download button |
| `plot_comparison()` | `processing.py` | Returns matplotlib figure for Streamlit |
| `plot_confusion_matrix()` | `processing.py` | Returns matplotlib figure for Streamlit |
| `plot_glcm_distributions()` | `processing.py` | Returns matplotlib figure for Streamlit |
| `plot_per_fold()` | `processing.py` | Returns matplotlib figure for Streamlit |

### Key Changes Made

1. **Image Input**: Changed from Kaggle API download to direct user upload
2. **Error Handling**: Added try-catch blocks for web robustness
3. **Progress Tracking**: Modified to use Streamlit progress indicators
4. **Visualization**: Functions now return matplotlib figures instead of saving to disk
5. **Session Management**: Used Streamlit session state to preserve data across interactions
6. **File I/O**: Uses temporary directories instead of fixed paths

---

## 🔧 Configuration

The `CONFIG` dictionary in `processing.py` controls model parameters:

```python
CONFIG = {
    'image_size': 224,              # Image resize dimension
    'clahe_clip': 2.0,             # CLAHE contrast limit
    'n_folds': 5,                  # Cross-validation folds
    'random_state': 42,            # For reproducibility
    'svm_param_grid': {...},       # SVM hyperparameters
    'ann_hidden': (100,),          # ANN architecture
    'class_names': ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
}
```

Modify these values in `processing.py` before running if needed.

---

## ⚠️ Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.10.0
```

If you have an M1/M2 Mac, use:
```bash
pip install tensorflow-macos
```

### Problem: "Error reading CSV file"

**Ensure your CSV:**
- Has columns named `id_code` and `diagnosis`
- Uses correct column names (case matters)
- Is a valid CSV format

### Problem: "Only X images uploaded but CSV has Y entries"

**Solution:**
- Make sure image filenames match the `id_code` values
- Include image file extensions (.png, .jpg)
- Filenames should be: `image_001.png`, not just `image_001`

### Problem: App is very slow

**Solutions:**
- Avoid training CNN (uncheck it)
- Use a smaller dataset for testing
- Run on a machine with more RAM
- Check if your CPU is being maxed out

### Problem: "WARNING: No images were successfully loaded"

**Ensure:**
- Image filenames exactly match CSV `id_code` values
- File extensions are .png or .jpg
- Images are valid, readable files
- No special characters in filenames

---

## 📈 Model Details

### SVM (Support Vector Machine)
- **Time**: ~5-10 minutes
- **Features**: GLCM (4 features)
- **Hyperparameter**: GridSearchCV on C and gamma
- **Best for**: Small to medium datasets

### ANN (Artificial Neural Network)
- **Time**: ~5-10 minutes
- **Architecture**: Input(4) → Hidden(100) → Output(5)
- **Activation**: ReLU with softmax output
- **Features**: Early stopping with validation split

### Fusion Framework
- **Time**: ~10-15 minutes
- **Method**: Weighted combination of SVM + ANN
- **Weights**: Computed from training F1 scores
- **Strength**: Combines strengths of both models

### CNN (Convolutional Neural Network)
- **Time**: 30+ minutes (requires GPU)
- **Architecture**: Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(256) → Output(5)
- **Input**: Full 224×224×3 RGB images
- **Recommended**: Train only if you have GPU and time

---

## 🎓 Dissertation Usage

This web application is designed to complement your dissertation:

1. **Data Collection**: Upload your own dataset or benchmark datasets
2. **Experimentation**: Train different model combinations
3. **Results Generation**: Automatically create publication-ready figures
4. **Statistical Analysis**: Compare models using Wilcoxon tests
5. **Documentation**: Export all results for your thesis

---

## 📚 References

**GLCM Features**: Haralick et al., "Textural Features for Image Classification"

**SVM**: Vapnik, "Statistical Learning Theory"

**ANN**: Goodfellow, Bengio, Courville, "Deep Learning" (MIT Press)

**Fusion Methods**: Kuncheva, "Combining Pattern Classifiers"

---

## 📧 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the code comments in `app.py` and `processing.py`
3. Consult your supervisor or course instructors

---

## 📄 License

This project is part of an MSc dissertation. Please respect intellectual property rights.

---

**Created**: 2026  
**Author**: Chandra Kiran Malkapuram  
**Institution**: University of East London  
**Supervisor**: Dr Fidal Bashir  

---

## ✨ Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare your data**: Organize CSV + images
3. **Run the app**: `streamlit run app.py`
4. **Upload dataset**: Use the Upload page
5. **Train models**: Select and train on Processing page
6. **Analyze results**: Explore visualizations and download outputs

Enjoy using the Diabetic Retinopathy Detection Web Application! 🏥
