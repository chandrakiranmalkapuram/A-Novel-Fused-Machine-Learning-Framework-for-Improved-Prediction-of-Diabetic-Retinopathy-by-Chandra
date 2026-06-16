"""
========================================================
  A Novel Fused Machine Learning Framework for
  Improved Prediction of Diabetic Retinopathy
  
  Author  : Chandra Kiran Malkapuram
  Student : U2904743
  MSc Cloud Computing - University of East London
  Supervisor: Dr Fidal Bashir
========================================================

HOW TO RUN:
  1. Open this folder in VS Code
  2. Open the terminal (View > Terminal)
  3. Run:  pip install -r requirements.txt
  4. Run:  python main.py
  5. Follow the on-screen menu

NOTE: First time setup will download the APTOS 2019 dataset
      automatically using your Kaggle credentials.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
import json
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
    'data_dir':        'data',
    'results_dir':     'results',
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

os.makedirs(CONFIG['data_dir'],    exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def banner(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def step(msg):
    print(f"\n  >> {msg}")

def done(msg="Done!"):
    print(f"     [OK] {msg}")


# ─────────────────────────────────────────────
#  STEP 1 — KAGGLE SETUP & DATASET DOWNLOAD
# ─────────────────────────────────────────────
def setup_kaggle():
    banner("STEP 1 — Kaggle Dataset Download")

    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json):
        print("""
  To download the APTOS 2019 dataset you need a free Kaggle account.

  Follow these steps:
  1. Go to https://www.kaggle.com and sign in (or create a free account)
  2. Click your profile picture (top right) → Settings
  3. Scroll to the API section → click 'Create New Token'
  4. A file called kaggle.json will download to your computer
  5. Copy that file into this folder: """ + kaggle_dir + """

  Then run this program again.
        """)
        os.makedirs(kaggle_dir, exist_ok=True)
        input("  Press Enter once you have placed kaggle.json in the folder above...")

    # Verify kaggle.json exists
    if not os.path.exists(kaggle_json):
        print("  ERROR: kaggle.json not found. Please follow the steps above.")
        sys.exit(1)

    # Set permissions
    os.chmod(kaggle_json, 0o600)

    # Check if dataset already downloaded
    csv_path = os.path.join(CONFIG['data_dir'], 'train.csv')
    img_dir  = os.path.join(CONFIG['data_dir'], 'train_images')

    if os.path.exists(csv_path) and os.path.exists(img_dir):
        n = len(os.listdir(img_dir))
        done(f"Dataset already downloaded ({n} images found)")
        return

    step("Downloading APTOS 2019 dataset from Kaggle...")
    print("     This may take 5-10 minutes depending on your internet speed.")

    ret = os.system(
        f'kaggle competitions download -c aptos2019-blindness-detection -p {CONFIG["data_dir"]}'
    )
    if ret != 0:
        print("\n  ERROR: Download failed. Make sure you have accepted the")
        print("  competition rules at: https://www.kaggle.com/c/aptos2019-blindness-detection")
        sys.exit(1)

    step("Extracting files...")
    zip_path = os.path.join(CONFIG['data_dir'], 'aptos2019-blindness-detection.zip')
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(CONFIG['data_dir'])
        os.remove(zip_path)

    done("Dataset ready!")


# ─────────────────────────────────────────────
#  STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_path):
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
#  STEP 3 — GLCM FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_glcm_features(image):
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


def load_and_extract(force=False):
    banner("STEP 2 — Loading Images & Extracting GLCM Features")

    feat_path = os.path.join(CONFIG['results_dir'], 'X_glcm.npy')
    label_path = os.path.join(CONFIG['results_dir'], 'y_labels.npy')

    if not force and os.path.exists(feat_path) and os.path.exists(label_path):
        step("Loading saved features...")
        X = np.load(feat_path)
        y = np.load(label_path)
        done(f"Loaded {X.shape[0]} samples, {X.shape[1]} features each")
        return X, y

    df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    img_dir = os.path.join(CONFIG['data_dir'], 'train_images')

    step(f"Extracting GLCM features from {len(df)} images...")
    print("     This takes about 10-15 minutes the first time. Please wait...")

    features_list, labels_list = [], []
    failed = 0
    start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        path = os.path.join(img_dir, row['id_code'] + '.png')
        img  = preprocess_image(path)
        if img is not None:
            features_list.append(extract_glcm_features(img))
            labels_list.append(row['diagnosis'])
        else:
            failed += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            pct = (i + 1) / len(df) * 100
            eta = (elapsed / (i + 1)) * (len(df) - i - 1)
            print(f"     Progress: {i+1}/{len(df)} ({pct:.0f}%) | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    X = np.array(features_list)
    y = np.array(labels_list)

    np.save(feat_path,  X)
    np.save(label_path, y)

    done(f"Extracted {X.shape[0]} samples | Failed: {failed} | "
         f"Time: {time.time()-start:.0f}s")
    return X, y


# ─────────────────────────────────────────────
#  STEP 4 — METRICS HELPER
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
    }


# ─────────────────────────────────────────────
#  STEP 5 — SVM
# ─────────────────────────────────────────────
def run_svm(X, y):
    banner("STEP 3 — Training SVM Classifier")
    skf    = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                             random_state=CONFIG['random_state'])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    all_scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
    best_model, best_f1 = None, 0

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        step(f"SVM Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr],    y[va]

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

        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("SVM", all_scores)
    return all_scores, best_model, scaler


# ─────────────────────────────────────────────
#  STEP 6 — ANN
# ─────────────────────────────────────────────
def run_ann(X, y, scaler):
    banner("STEP 4 — Training ANN Classifier")
    skf  = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                           random_state=CONFIG['random_state'])
    X_sc = scaler.transform(X)

    all_scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
    best_model, best_f1 = None, 0

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        step(f"ANN Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr],    y[va]

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

        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("ANN", all_scores)
    return all_scores, best_model


# ─────────────────────────────────────────────
#  STEP 7 — FUZZY FUSION
# ─────────────────────────────────────────────
def run_fusion(X, y, scaler):
    banner("STEP 5 — Training Fuzzy Fusion Framework")
    skf  = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                           random_state=CONFIG['random_state'])
    X_sc = scaler.transform(X)

    all_scores  = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
    last_preds  = None
    last_true   = None
    last_probs  = None

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        step(f"Fusion Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr],    y[va]

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
        w2 = f1_score(ytr, ann.predict(Xtr),     average='macro', zero_division=0)
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
        last_true  = yva
        last_probs = fused_probs

        print(f"     w_SVM={w1:.3f} | w_ANN={w2:.3f} | "
              f"Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("Fusion", all_scores)
    return all_scores, last_true, last_preds, last_probs


# ─────────────────────────────────────────────
#  STEP 8 — CNN BASELINE
# ─────────────────────────────────────────────
def build_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32,  (3,3), activation='relu',
                      input_shape=(CONFIG['image_size'], CONFIG['image_size'], 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64,  (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3,3), activation='relu'),
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


def run_cnn(df_path, img_dir):
    banner("STEP 6 — Training CNN Baseline")
    from tensorflow.keras.callbacks import EarlyStopping

    step("Loading images for CNN (this takes several minutes)...")
    df  = pd.read_csv(df_path)
    imgs, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(img_dir, row['id_code'] + '.png')
        img  = preprocess_image(path)
        if img is not None:
            imgs.append(img)
            labels.append(row['diagnosis'])

    X_img = np.array(imgs,   dtype=np.float32)
    y_img = np.array(labels, dtype=np.int32)
    done(f"Loaded {len(X_img)} images")

    skf        = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                                 random_state=CONFIG['random_state'])
    all_scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}

    for fold, (tr, va) in enumerate(skf.split(X_img, y_img)):
        step(f"CNN Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_img[tr], X_img[va]
        ytr, yva = y_img[tr], y_img[va]

        model = build_cnn()
        es    = EarlyStopping(patience=CONFIG['cnn_patience'],
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

        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("CNN", all_scores)
    return all_scores


# ─────────────────────────────────────────────
#  STEP 9 — PRINT FOLD SUMMARY
# ─────────────────────────────────────────────
def print_fold_summary(name, scores):
    print(f"\n  {'─'*40}")
    print(f"  {name} — 5-Fold Summary:")
    for metric, vals in scores.items():
        print(f"    {metric.upper():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  {'─'*40}")


# ─────────────────────────────────────────────
#  STEP 10 — RESULTS TABLE + WILCOXON
# ─────────────────────────────────────────────
def print_results_table(all_results):
    banner("FINAL RESULTS TABLE")
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    names   = list(all_results.keys())

    header = f"  {'Metric':<12}" + "".join(f"  {n:>18}" for n in names)
    print(header)
    print("  " + "─" * (12 + 20 * len(names)))

    for m in metrics:
        row = f"  {m.upper():<12}"
        for n in names:
            mean = np.mean(all_results[n][m])
            std  = np.std(all_results[n][m])
            row += f"  {mean:.4f} ± {std:.4f}"
        print(row)

    # Wilcoxon test: Fusion vs CNN
    if 'Fusion' in all_results and 'CNN' in all_results:
        print("\n  Wilcoxon Signed-Rank Test (Fusion vs CNN):")
        for m in metrics:
            try:
                stat, p = wilcoxon(all_results['Fusion'][m], all_results['CNN'][m])
                sig = "SIGNIFICANT ✓" if p < 0.05 else "not significant"
                print(f"    {m.upper():<12}: p = {p:.4f} → {sig}")
            except Exception:
                print(f"    {m.upper():<12}: insufficient data for test")


# ─────────────────────────────────────────────
#  STEP 11 — SAVE RESULTS CSV
# ─────────────────────────────────────────────
def save_results_csv(all_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    rows = []
    for name, scores in all_results.items():
        row = {'Model': name}
        for m in metrics:
            row[f'{m}_mean'] = round(np.mean(scores[m]), 4)
            row[f'{m}_std']  = round(np.std(scores[m]),  4)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(CONFIG['results_dir'], 'results_summary.csv')
    df.to_csv(path, index=False)
    done(f"Results saved to {path}")
    return df


# ─────────────────────────────────────────────
#  STEP 12 — PLOTS
# ─────────────────────────────────────────────
def plot_comparison(all_results):
    banner("Generating Comparison Charts")
    metrics     = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(all_results.keys())
    colors      = CONFIG['colors'][:len(model_names)]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Model Performance Comparison — 5-Fold Cross-Validation',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, metric in enumerate(metrics):
        means = [np.mean(all_results[m][metric]) for m in model_names]
        stds  = [np.std(all_results[m][metric])  for m in model_names]
        bars  = axes[i].bar(model_names, means, yerr=stds, color=colors,
                            capsize=5, edgecolor='white', linewidth=0.5)
        axes[i].set_title(metric.upper(), fontsize=11, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=15)
        for j, (m, s) in enumerate(zip(means, stds)):
            axes[i].text(j, m + s + 0.01, f'{m:.3f}',
                         ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'comparison_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved to {path}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG['class_names'],
                yticklabels=CONFIG['class_names'],
                linewidths=0.5)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
    plt.ylabel('True Label',      fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    path  = os.path.join(CONFIG['results_dir'], fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved to {path}")


def plot_glcm_distributions(X, y):
    banner("Generating GLCM Feature Distribution Plots")
    feature_names  = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
    class_colors   = ['#7F77DD','#1D9E75','#D85A30','#378ADD','#EF9F27']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('GLCM Feature Distributions by DR Severity Grade',
                 fontsize=13, fontweight='bold')

    for i, feat in enumerate(feature_names):
        for c in range(5):
            vals = X[y == c, i]
            axes[i].hist(vals, alpha=0.6, label=CONFIG['class_names'][c],
                         color=class_colors[c], bins=25, edgecolor='none')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xlabel('Feature Value')
        axes[i].set_ylabel('Frequency')
        if i == 0:
            axes[i].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'glcm_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved to {path}")


def plot_per_fold(all_results):
    banner("Generating Per-Fold Performance Plots")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Fold F1 Score — All Models', fontsize=13, fontweight='bold')

    model_names = list(all_results.keys())
    colors      = CONFIG['colors'][:len(model_names)]
    folds       = list(range(1, CONFIG['n_folds'] + 1))

    for ax, metric in zip(axes.flat, ['accuracy', 'f1', 'recall', 'auc']):
        for name, col in zip(model_names, colors):
            vals = all_results[name][metric]
            ax.plot(folds, vals, marker='o', label=name, color=col, linewidth=2)
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'per_fold_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved to {path}")


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   Diabetic Retinopathy Detection — MSc Dissertation      ║
║   Chandra Kiran Malkapuram | U2904743 | UEL              ║
╚══════════════════════════════════════════════════════════╝

  Select an option:

  [1]  Full Pipeline  (runs everything — takes 1-2 hours)
  [2]  GLCM + SVM + ANN + Fusion only  (no CNN, faster ~30 min)
  [3]  Load saved results and generate charts only
  [4]  Exit
    """)

    choice = input("  Enter choice (1/2/3/4): ").strip()

    if choice == '4':
        print("  Goodbye!")
        sys.exit(0)

    # ── Setup dataset ──────────────────────────────────────
    setup_kaggle()

    df_path = os.path.join(CONFIG['data_dir'], 'train.csv')
    img_dir = os.path.join(CONFIG['data_dir'], 'train_images')

    all_results = {}

    if choice == '3':
        # Load from saved CSV
        csv = os.path.join(CONFIG['results_dir'], 'results_summary.csv')
        if not os.path.exists(csv):
            print("  No saved results found. Please run option 1 or 2 first.")
            sys.exit(1)
        df_res = pd.read_csv(csv)
        print(f"\n  Loaded results from {csv}")
        print(df_res.to_string(index=False))
        print("\n  Charts cannot be regenerated without running the models.")
        print("  Please check the results/ folder for existing charts.")
        sys.exit(0)

    # ── Extract features ────────────────────────────────────
    X_glcm, y = load_and_extract()

    # ── SVM ─────────────────────────────────────────────────
    svm_scores, best_svm, scaler = run_svm(X_glcm, y)
    all_results['SVM'] = svm_scores

    # ── ANN ─────────────────────────────────────────────────
    ann_scores, best_ann = run_ann(X_glcm, y, scaler)
    all_results['ANN'] = ann_scores

    # ── Fusion ──────────────────────────────────────────────
    fusion_scores, f_true, f_pred, f_prob = run_fusion(X_glcm, y, scaler)
    all_results['Fusion'] = fusion_scores

    # ── CNN (optional) ──────────────────────────────────────
    if choice == '1':
        cnn_scores = run_cnn(df_path, img_dir)
        all_results['CNN'] = cnn_scores

    # ── Results ─────────────────────────────────────────────
    print_results_table(all_results)
    save_results_csv(all_results)

    # ── Plots ───────────────────────────────────────────────
    banner("Generating All Figures")
    plot_comparison(all_results)
    plot_confusion_matrix(f_true, f_pred, "Fusion Framework")
    plot_glcm_distributions(X_glcm, y)
    plot_per_fold(all_results)

    # ── Done ────────────────────────────────────────────────
    banner("ALL DONE!")
    print(f"""
  Your results have been saved to the 'results/' folder:

    results_summary.csv        — All metrics (mean ± std)
    comparison_chart.png       — Bar chart comparing all models
    fusion_confusion_matrix.png — Confusion matrix for fusion model
    glcm_distributions.png     — GLCM feature distributions by DR grade
    per_fold_performance.png   — Per-fold line plots

  You can now use these figures and numbers directly in Chapter 4
  of your dissertation!
    """)


if __name__ == '__main__':
    main()
