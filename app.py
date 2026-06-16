"""
========================================================
  Diabetic Retinopathy Detection Web Application
  
  A modern Streamlit web app for DR prediction using
  machine learning classification with SVM, ANN, and
  Fuzzy Fusion frameworks.
========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from processing import (
    CONFIG, extract_features_from_images, compute_summary_stats,
    run_svm, run_ann, run_fusion, run_cnn,
    plot_comparison_chart, plot_confusion_matrix, plot_glcm_distributions,
    plot_per_fold, wilcoxon_test
)

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DR Detection Web App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77d4;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader-info {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


def save_uploaded_files(uploaded_files, destination_dir):
    """Save uploaded files to destination directory."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(destination_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def validate_dataset(csv_file, image_files):
    """Validate that CSV and images are compatible."""
    errors = []
    
    # Check if CSV is provided
    if csv_file is None:
        errors.append("❌ CSV file is required")
        return errors
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        errors.append(f"❌ Error reading CSV file: {str(e)}")
        return errors
    
    # Check required columns
    if 'id_code' not in df.columns and 'id_code' not in [col.lower() for col in df.columns]:
        errors.append("❌ CSV must have an 'id_code' column")
    
    if 'diagnosis' not in df.columns and 'diagnosis' not in [col.lower() for col in df.columns]:
        errors.append("❌ CSV must have a 'diagnosis' column")
    
    # Check image count
    if len(image_files) == 0:
        errors.append("❌ At least one image is required")
    elif len(image_files) < len(df):
        errors.append(
            f"⚠️  Only {len(image_files)} images uploaded but CSV has "
            f"{len(df)} entries. Missing images will be skipped."
        )
    
    return errors


def download_file(file_path, file_name):
    """Create a download button for a file."""
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    return st.download_button(
        label=f"📥 Download {file_name}",
        data=file_data,
        file_name=file_name,
        mime="application/octet-stream",
        key=file_name
    )


# ─────────────────────────────────────────────
#  SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'dataset_ready' not in st.session_state:
        st.session_state.dataset_ready = False
    if 'X_features' not in st.session_state:
        st.session_state.X_features = None
    if 'y_labels' not in st.session_state:
        st.session_state.y_labels = None
    if 'image_ids' not in st.session_state:
        st.session_state.image_ids = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'fusion_predictions' not in st.session_state:
        st.session_state.fusion_predictions = None


init_session_state()
ensure_directories()


# ─────────────────────────────────────────────
#  MAIN APP LAYOUT
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">🏥 Diabetic Retinopathy Detection</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="subheader-info">A Modern ML Framework for DR Prediction</p>', 
            unsafe_allow_html=True)

st.divider()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio(
    "Select Page:",
    ["📤 Upload Dataset", "🔬 Processing", "📈 Results & Analysis"],
    index=0
)

st.sidebar.divider()
st.sidebar.info(
    "**About this App**\n\n"
    "This web application provides a complete machine learning pipeline "
    "for Diabetic Retinopathy detection using:\n"
    "- **SVM**: Support Vector Machine classifier\n"
    "- **ANN**: Artificial Neural Network\n"
    "- **Fusion**: Fuzzy fusion of SVM and ANN\n"
    "- **CNN**: Convolutional Neural Network baseline\n\n"
    "Upload your dataset and click Process to start!"
)


# ─────────────────────────────────────────────
#  PAGE 1: UPLOAD DATASET
# ─────────────────────────────────────────────
if app_mode == "📤 Upload Dataset":
    st.header("📤 Upload Your Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 CSV File")
        st.info(
            "**Required columns:**\n"
            "- `id_code`: Image ID (without extension)\n"
            "- `diagnosis`: DR grade (0-4)"
        )
        csv_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            key='csv_upload',
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("🖼️ Image Files")
        st.info(
            "Upload retinal fundus images (PNG/JPG).\n"
            "Image names should match the `id_code` in your CSV."
        )
        image_files = st.file_uploader(
            "Upload images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key='image_upload',
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Validation
    if csv_file is not None or image_files:
        st.subheader("✅ Validation")
        errors = validate_dataset(csv_file, image_files)
        
        if errors:
            for error in errors:
                st.warning(error)
        else:
            st.success("✅ All validations passed!")
        
        # Show preview
        if csv_file is not None:
            df_preview = pd.read_csv(csv_file)
            st.subheader("📊 CSV Preview")
            st.dataframe(df_preview.head(10), use_container_width=True)
            st.caption(f"Total rows: {len(df_preview)}")
        
        if image_files:
            st.subheader(f"📸 Uploaded Images ({len(image_files)})")
            cols = st.columns(5)
            for idx, img_file in enumerate(image_files[:5]):
                with cols[idx % 5]:
                    st.image(img_file, use_column_width=True)
            if len(image_files) > 5:
                st.caption(f"+ {len(image_files) - 5} more images")
        
        # Process button
        st.divider()
        if st.button("🚀 Load Dataset for Processing", use_container_width=True, 
                     type="primary", key="load_dataset"):
            
            with st.spinner("Loading and extracting features..."):
                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save files
                    csv_path = os.path.join(temp_dir, 'data.csv')
                    image_dir = os.path.join(temp_dir, 'images')
                    os.makedirs(image_dir, exist_ok=True)
                    
                    with open(csv_path, 'wb') as f:
                        f.write(csv_file.getbuffer())
                    
                    for img_file in image_files:
                        img_path = os.path.join(image_dir, img_file.name)
                        with open(img_path, 'wb') as f:
                            f.write(img_file.getbuffer())
                    
                    # Read CSV
                    df = pd.read_csv(csv_path)
                    
                    # Normalize column names
                    df.columns = [col.lower() for col in df.columns]
                    
                    image_ids = df['id_code'].values
                    labels = df['diagnosis'].values
                    
                    # Extract features
                    progress_bar = st.progress(0)
                    def progress_callback(current, total):
                        progress_bar.progress(min(current / total, 1.0))
                    
                    X, failed_ids = extract_features_from_images(
                        image_dir, image_ids, progress_callback
                    )
                    
                    # Store in session
                    st.session_state.X_features = X
                    st.session_state.y_labels = labels[:len(X)]  # Match with extracted features
                    st.session_state.image_ids = image_ids[:len(X)]
                    st.session_state.dataset_ready = True
                    
                    # Clean up
                    shutil.rmtree(temp_dir)
                    
                    st.success(
                        f"✅ Dataset loaded successfully!\n\n"
                        f"- Extracted {len(X)} feature sets\n"
                        f"- Features per sample: {X.shape[1]}\n"
                        f"- Failed to load: {len(failed_ids)} images"
                    )
                    
                    st.info("👉 Go to the **Processing** page to train models")
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")


# ─────────────────────────────────────────────
#  PAGE 2: PROCESSING
# ─────────────────────────────────────────────
elif app_mode == "🔬 Processing":
    st.header("🔬 Model Training & Processing")
    
    if not st.session_state.dataset_ready:
        st.warning("⚠️  Please upload and load a dataset first on the **Upload Dataset** page")
    else:
        st.success(f"✅ Dataset ready: {len(st.session_state.X_features)} samples loaded")
        
        st.divider()
        st.subheader("🤖 Select Models to Train")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            train_svm = st.checkbox("SVM", value=True)
        with col2:
            train_ann = st.checkbox("ANN", value=True)
        with col3:
            train_fusion = st.checkbox("Fusion", value=True)
        with col4:
            train_cnn = st.checkbox("CNN", value=False, 
                                   help="CNN takes much longer to train")
        
        st.divider()
        
        # Training settings
        st.subheader("⚙️  Training Settings")
        n_folds = st.slider("Number of Folds", 3, 10, 5)
        
        st.divider()
        
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            st.session_state.results = {}
            
            progress_container = st.container()
            
            try:
                # SVM
                if train_svm:
                    with progress_container:
                        st.info("🔄 Training SVM...")
                        svm_scores, best_svm, scaler, svm_folds = run_svm(
                            st.session_state.X_features,
                            st.session_state.y_labels
                        )
                        st.session_state.results['SVM'] = svm_scores
                        st.session_state.scaler = scaler
                        st.success("✅ SVM training complete!")
                
                # ANN
                if train_ann:
                    with progress_container:
                        st.info("🔄 Training ANN...")
                        ann_scores, best_ann, ann_folds = run_ann(
                            st.session_state.X_features,
                            st.session_state.y_labels,
                            st.session_state.scaler
                        )
                        st.session_state.results['ANN'] = ann_scores
                        st.success("✅ ANN training complete!")
                
                # Fusion
                if train_fusion:
                    with progress_container:
                        st.info("🔄 Training Fusion Framework...")
                        fusion_scores, f_true, f_pred, f_prob, fusion_folds = run_fusion(
                            st.session_state.X_features,
                            st.session_state.y_labels,
                            st.session_state.scaler
                        )
                        st.session_state.results['Fusion'] = fusion_scores
                        st.session_state.fusion_predictions = (f_true, f_pred, f_prob)
                        st.success("✅ Fusion training complete!")
                
                # CNN
                if train_cnn:
                    st.warning("⚠️  CNN training requires images. This feature works "
                             "differently in the web app.")
                
                st.divider()
                st.success("🎉 All selected models trained successfully!")
                st.info("👉 Go to the **Results & Analysis** page to view metrics, "
                       "visualizations, and download results")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                import traceback
                st.text(traceback.format_exc())


# ─────────────────────────────────────────────
#  PAGE 3: RESULTS & ANALYSIS
# ─────────────────────────────────────────────
elif app_mode == "📈 Results & Analysis":
    st.header("📈 Results & Analysis")
    
    if not st.session_state.results:
        st.warning("⚠️  No training results available. Please train models first on the "
                  "**Processing** page")
    else:
        # Summary Statistics
        st.subheader("📊 Summary Statistics")
        
        summary_df = compute_summary_stats(st.session_state.results)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download summary
        col1, col2 = st.columns([3, 1])
        with col2:
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="results_summary.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Comparison Chart",
            "🔥 Confusion Matrix",
            "📉 GLCM Distributions",
            "📋 Per-Fold Performance"
        ])
        
        with tab1:
            st.write("Model performance comparison across all metrics")
            try:
                fig = plot_comparison_chart(st.session_state.results)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
        
        with tab2:
            st.write("Confusion matrix for Fusion model")
            if st.session_state.fusion_predictions is not None:
                try:
                    f_true, f_pred, _ = st.session_state.fusion_predictions
                    fig = plot_confusion_matrix(f_true, f_pred, "Fusion Framework")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating confusion matrix: {str(e)}")
            else:
                st.info("Fusion model not trained. Train it in the Processing page.")
        
        with tab3:
            st.write("GLCM feature distributions by DR severity grade")
            try:
                fig = plot_glcm_distributions(
                    st.session_state.X_features,
                    st.session_state.y_labels
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating distributions: {str(e)}")
        
        with tab4:
            st.write("Performance metrics for each fold")
            try:
                fig = plot_per_fold(st.session_state.results)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating per-fold plot: {str(e)}")
        
        st.divider()
        
        # Statistical Tests
        st.subheader("📊 Statistical Analysis")
        
        if 'Fusion' in st.session_state.results and 'CNN' not in st.session_state.results:
            st.info("Wilcoxon test requires at least two models")
        
        st.divider()
        
        # Detailed Results by Model
        st.subheader("🔍 Detailed Results by Model")
        
        model_choice = st.selectbox(
            "Select a model to view fold details",
            list(st.session_state.results.keys())
        )
        
        if model_choice:
            scores = st.session_state.results[model_choice]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            metrics_data = {
                'accuracy': np.mean(scores['accuracy']),
                'precision': np.mean(scores['precision']),
                'recall': np.mean(scores['recall']),
                'f1': np.mean(scores['f1']),
                'auc': np.mean(scores['auc'])
            }
            
            with col1:
                st.metric("Accuracy", f"{metrics_data['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics_data['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics_data['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics_data['f1']:.4f}")
            with col5:
                st.metric("AUC", f"{metrics_data['auc']:.4f}")
            
            # Per-fold details
            fold_details = []
            for fold in range(len(scores['accuracy'])):
                fold_details.append({
                    'Fold': fold + 1,
                    'Accuracy': f"{scores['accuracy'][fold]:.4f}",
                    'Precision': f"{scores['precision'][fold]:.4f}",
                    'Recall': f"{scores['recall'][fold]:.4f}",
                    'F1': f"{scores['f1'][fold]:.4f}",
                    'AUC': f"{scores['auc'][fold]:.4f}"
                })
            
            fold_df = pd.DataFrame(fold_details)
            st.dataframe(fold_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
---
**Diabetic Retinopathy Detection Web Application**

MSc Dissertation | Chandra Kiran Malkapuram

This application implements a novel fused machine learning framework for 
improved prediction of Diabetic Retinopathy using GLCM features and ensemble methods.
""", unsafe_allow_html=True)
