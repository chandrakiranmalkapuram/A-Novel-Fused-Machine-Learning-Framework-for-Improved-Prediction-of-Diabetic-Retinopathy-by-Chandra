"""
========================================================
  Processing Module for Diabetic Retinopathy Detection
  
  This module contains all the data processing and
  machine learning logic extracted from main.py
  for use in the Streamlit web application.
========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import wilcoxon


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'image_size':      224,
    'clahe_clip':      2.0,
    'clahe_tile':      (8, 8),
    'glcm_distances':  [1],
    'glcm_angles':     [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'glcm_levels':     256,
    'n_folds':         5,
    'random_state':    42,
    'svm_param_grid':  {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
    'ann_hidden':      (100,),
    'ann_max_iter':    200,
    'ann_alpha':       0.0001,
    'cnn_epochs':      30,
    'cnn_batch':       32,
    'cnn_patience':    5,
    'class_names':     ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
    'colors':          ['#7F77DD', '#1D9E75', '#D85A30', '#378ADD'],
}


# ─────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    """
    Preprocess a single image:
    - Resize to 224x224
    - Apply CLAHE on green channel
    - Normalize to [0, 1]
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize with bicubic interpolation
    img = cv2.resize(img, (CONFIG['image_size'], CONFIG['image_size']),
                     interpolation=cv2.INTER_CUBIC)
    
    # CLAHE on green channel
    clahe = cv2.createCLAHE(clipLimit=CONFIG['clahe_clip'],
                             tileGridSize=CONFIG['clahe_tile'])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    
    # Normalise to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


# ─────────────────────────────────────────────
#  GLCM FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_glcm_features(image):
    """
    Extract GLCM (Gray Level Co-occurrence Matrix) features:
    - Contrast
    - Correlation
    - Energy
    - Homogeneity
    """
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray,
        distances=CONFIG['glcm_distances'],
        angles=CONFIG['glcm_angles'],
        levels=CONFIG['glcm_levels'],
        symmetric=True,
        normed=True
    )
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.append(np.mean(graycoprops(glcm, prop)[0]))
    return np.array(features)


def extract_features_from_images(image_dir, image_ids, progress_callback=None):
    """
    Extract GLCM features from a directory of images.
    
    Args:
        image_dir: Directory containing image files
        image_ids: List of image IDs (without extension)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        failed_ids: List of image IDs that failed to process
    """
    features_list = []
    failed_ids = []
    
    for idx, image_id in enumerate(image_ids):
        image_path = os.path.join(image_dir, image_id + '.png')
        
        if not os.path.exists(image_path):
            failed_ids.append(image_id)
            continue
        
        img = preprocess_image(image_path)
        if img is not None:
            try:
                features = extract_glcm_features(img)
                features_list.append(features)
            except Exception as e:
                failed_ids.append(image_id)
        else:
            failed_ids.append(image_id)
        
        if progress_callback and (idx + 1) % 10 == 0:
            progress_callback(idx + 1, len(image_ids))
    
    X = np.array(features_list) if features_list else np.array([]).reshape(0, 4)
    return X, failed_ids


# ─────────────────────────────────────────────
#  METRICS COMPUTATION
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics.
    """
    try:
        return {
            'accuracy':  float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall':    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1':        float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'auc':       float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')),
        }
    except Exception as e:
        # Return zero metrics if computation fails
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
        }


# ─────────────────────────────────────────────
#  SVM CLASSIFIER
# ─────────────────────────────────────────────
def run_svm(X, y, progress_callback=None):
    """
    Train SVM classifier with 5-fold cross-validation.
    
    Returns:
        all_scores: Dictionary of metrics for each fold
        best_model: Best trained SVM model
        scaler: Fitted StandardScaler
    """
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                          random_state=CONFIG['random_state'])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    all_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    best_model, best_f1 = None, 0
    fold_details = []

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr], y[va]

        gs = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=CONFIG['random_state']),
            CONFIG['svm_param_grid'], cv=3, scoring='f1_macro', n_jobs=-1
        )
        gs.fit(Xtr, ytr)
        clf = gs.best_estimator_

        y_pred = clf.predict(Xva)
        y_prob = clf.predict_proba(Xva)
        m = compute_metrics(yva, y_pred, y_prob)
        
        for k, v in m.items():
            all_scores[k].append(v)

        if m['f1'] > best_f1:
            best_f1, best_model = m['f1'], clf

        fold_details.append({
            'fold': fold + 1,
            'accuracy': m['accuracy'],
            'f1': m['f1'],
            'auc': m['auc']
        })
        
        if progress_callback:
            progress_callback(f"SVM Fold {fold+1}/{CONFIG['n_folds']}")

    return all_scores, best_model, scaler, fold_details


# ─────────────────────────────────────────────
#  ANN CLASSIFIER
# ─────────────────────────────────────────────
def run_ann(X, y, scaler, progress_callback=None):
    """
    Train ANN classifier with 5-fold cross-validation.
    
    Returns:
        all_scores: Dictionary of metrics for each fold
        best_model: Best trained ANN model
    """
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                          random_state=CONFIG['random_state'])
    X_sc = scaler.transform(X)

    all_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    best_model, best_f1 = None, 0
    fold_details = []

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr], y[va]

        ann = MLPClassifier(
            hidden_layer_sizes=CONFIG['ann_hidden'],
            activation='relu', solver='adam',
            alpha=CONFIG['ann_alpha'], batch_size=32,
            max_iter=CONFIG['ann_max_iter'],
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=CONFIG['random_state']
        )
        ann.fit(Xtr, ytr)

        y_pred = ann.predict(Xva)
        y_prob = ann.predict_proba(Xva)
        m = compute_metrics(yva, y_pred, y_prob)
        
        for k, v in m.items():
            all_scores[k].append(v)

        if m['f1'] > best_f1:
            best_f1, best_model = m['f1'], ann

        fold_details.append({
            'fold': fold + 1,
            'accuracy': m['accuracy'],
            'f1': m['f1'],
            'auc': m['auc']
        })
        
        if progress_callback:
            progress_callback(f"ANN Fold {fold+1}/{CONFIG['n_folds']}")

    return all_scores, best_model, fold_details


# ─────────────────────────────────────────────
#  FUZZY FUSION FRAMEWORK
# ─────────────────────────────────────────────
def run_fusion(X, y, scaler, progress_callback=None):
    """
    Train Fuzzy Fusion framework combining SVM and ANN.
    
    Returns:
        all_scores: Dictionary of metrics for each fold
        fusion_predictions: Final predictions for last fold
        fusion_true_labels: True labels for last fold
        fusion_probabilities: Fused probabilities for last fold
    """
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                          random_state=CONFIG['random_state'])
    X_sc = scaler.transform(X)

    all_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    last_preds = None
    last_true = None
    last_probs = None
    fold_details = []

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr], y[va]

        # Train SVM
        gs = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=CONFIG['random_state']),
            CONFIG['svm_param_grid'], cv=3, scoring='f1_macro', n_jobs=-1
        )
        gs.fit(Xtr, ytr)
        svm_clf = gs.best_estimator_

        # Train ANN
        ann = MLPClassifier(
            hidden_layer_sizes=CONFIG['ann_hidden'],
            activation='relu', solver='adam',
            alpha=CONFIG['ann_alpha'], max_iter=CONFIG['ann_max_iter'],
            early_stopping=True, random_state=CONFIG['random_state']
        )
        ann.fit(Xtr, ytr)

        # Compute weights from training F1
        w1 = f1_score(ytr, svm_clf.predict(Xtr), average='macro', zero_division=0)
        w2 = f1_score(ytr, ann.predict(Xtr), average='macro', zero_division=0)
        total = w1 + w2 + 1e-9
        w1, w2 = w1 / total, w2 / total

        # Fused probabilities
        p1 = svm_clf.predict_proba(Xva)
        p2 = ann.predict_proba(Xva)
        fused_probs = (w1 * p1) + (w2 * p2)
        y_pred = np.argmax(fused_probs, axis=1)

        m = compute_metrics(yva, y_pred, fused_probs)
        for k, v in m.items():
            all_scores[k].append(v)

        last_preds = y_pred
        last_true = yva
        last_probs = fused_probs

        fold_details.append({
            'fold': fold + 1,
            'w_svm': float(w1),
            'w_ann': float(w2),
            'accuracy': m['accuracy'],
            'f1': m['f1'],
            'auc': m['auc']
        })
        
        if progress_callback:
            progress_callback(f"Fusion Fold {fold+1}/{CONFIG['n_folds']}")

    return all_scores, last_true, last_preds, last_probs, fold_details


# ─────────────────────────────────────────────
#  CNN BASELINE
# ─────────────────────────────────────────────
def build_cnn():
    """Build a simple CNN model."""
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(CONFIG['image_size'], CONFIG['image_size'], 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_cnn(image_dir, image_ids, labels, progress_callback=None):
    """
    Train CNN baseline with 5-fold cross-validation.
    """
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Load images
    imgs = []
    valid_labels = []
    valid_ids = []
    
    for image_id, label in zip(image_ids, labels):
        image_path = os.path.join(image_dir, image_id + '.png')
        if os.path.exists(image_path):
            img = preprocess_image(image_path)
            if img is not None:
                imgs.append(img)
                valid_labels.append(label)
                valid_ids.append(image_id)
    
    X_img = np.array(imgs, dtype=np.float32)
    y_img = np.array(valid_labels, dtype=np.int32)

    if len(X_img) == 0:
        return {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}, []

    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                          random_state=CONFIG['random_state'])
    all_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
    fold_details = []

    for fold, (tr, va) in enumerate(skf.split(X_img, y_img)):
        Xtr, Xva = X_img[tr], X_img[va]
        ytr, yva = y_img[tr], y_img[va]

        model = build_cnn()
        es = EarlyStopping(patience=CONFIG['cnn_patience'],
                           restore_best_weights=True, verbose=0)
        model.fit(Xtr, ytr,
                  epochs=CONFIG['cnn_epochs'],
                  batch_size=CONFIG['cnn_batch'],
                  validation_split=0.1,
                  callbacks=[es], verbose=0)

        y_prob = model.predict(Xva, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        m = compute_metrics(yva, y_pred, y_prob)
        
        for k, v in m.items():
            all_scores[k].append(v)

        fold_details.append({
            'fold': fold + 1,
            'accuracy': m['accuracy'],
            'f1': m['f1'],
            'auc': m['auc']
        })
        
        if progress_callback:
            progress_callback(f"CNN Fold {fold+1}/{CONFIG['n_folds']}")

    return all_scores, fold_details


# ─────────────────────────────────────────────
#  RESULTS ANALYSIS
# ─────────────────────────────────────────────
def compute_summary_stats(all_results):
    """
    Compute summary statistics for all models.
    
    Returns a DataFrame with mean ± std for all metrics.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    rows = []
    
    for name, scores in all_results.items():
        row = {'Model': name}
        for m in metrics:
            mean = np.mean(scores[m]) if scores[m] else 0.0
            std = np.std(scores[m]) if scores[m] else 0.0
            row[f'{m}_mean'] = round(mean, 4)
            row[f'{m}_std'] = round(std, 4)
        rows.append(row)
    
    return pd.DataFrame(rows)


def wilcoxon_test(all_results, model1='Fusion', model2='CNN'):
    """
    Perform Wilcoxon signed-rank test between two models.
    """
    if model1 not in all_results or model2 not in all_results:
        return None
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    results = {}
    
    for m in metrics:
        try:
            stat, p_value = wilcoxon(all_results[model1][m], all_results[model2][m])
            results[m] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        except Exception:
            results[m] = {
                'statistic': None,
                'p_value': None,
                'significant': False
            }
    
    return results


# ─────────────────────────────────────────────
#  VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────
def plot_comparison_chart(all_results):
    """
    Generate a comparison chart for all models.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(all_results.keys())
    colors = CONFIG['colors'][:len(model_names)]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Model Performance Comparison — 5-Fold Cross-Validation',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, metric in enumerate(metrics):
        means = [np.mean(all_results[m][metric]) for m in model_names]
        stds = [np.std(all_results[m][metric]) for m in model_names]
        bars = axes[i].bar(model_names, means, yerr=stds, color=colors,
                           capsize=5, edgecolor='white', linewidth=0.5)
        axes[i].set_title(metric.upper(), fontsize=11, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=15)
        
        for j, (m, s) in enumerate(zip(means, stds)):
            axes[i].text(j, m + s + 0.01, f'{m:.3f}',
                         ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Generate confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'],
                linewidths=0.5, ax=ax)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    
    return fig


def plot_glcm_distributions(X, y):
    """
    Plot GLCM feature distributions by DR grade.
    """
    feature_names = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
    class_colors = ['#7F77DD', '#1D9E75', '#D85A30', '#378ADD', '#EF9F27']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('GLCM Feature Distributions by DR Severity Grade',
                 fontsize=13, fontweight='bold')

    for i, feat in enumerate(feature_names):
        for c in range(min(5, len(np.unique(y)))):
            vals = X[y == c, i]
            if len(vals) > 0:
                axes[i].hist(vals, alpha=0.6, label=CONFIG['class_names'][c],
                             color=class_colors[c], bins=25, edgecolor='none')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xlabel('Feature Value')
        axes[i].set_ylabel('Frequency')
        if i == 0:
            axes[i].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_per_fold(all_results):
    """
    Plot per-fold performance metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Fold Performance — All Models', fontsize=13, fontweight='bold')

    model_names = list(all_results.keys())
    colors = CONFIG['colors'][:len(model_names)]
    folds = list(range(1, CONFIG['n_folds'] + 1))

    for ax, metric in zip(axes.flat, ['accuracy', 'f1', 'recall', 'auc']):
        for name, col in zip(model_names, colors):
            vals = all_results[name][metric]
            if vals:
                ax.plot(folds, vals, marker='o', label=name, color=col, linewidth=2)
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
