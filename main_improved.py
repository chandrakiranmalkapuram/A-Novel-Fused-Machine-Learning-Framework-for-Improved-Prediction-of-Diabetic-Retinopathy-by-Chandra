"""
========================================================
  IMPROVED DR Detection Pipeline
  EfficientNet Transfer Learning + SMOTE

  Author  : Chandra Kiran Malkapuram
  Student : U2904743
  MSc Cloud Computing - University of East London
  Supervisor: Dr Fidal Bashir

  CHANGES FROM ORIGINAL:
  1. CNN replaced with EfficientNetB0 (transfer learning)
  2. SMOTE applied to balance minority DR classes
  3. Extended GLCM features (GLCM + LBP combined)
========================================================
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
warnings.filterwarnings('ignore')

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from scipy.stats import wilcoxon

# ── SMOTE for class balancing ────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Installing imbalanced-learn for SMOTE...")
    os.system("pip install imbalanced-learn -q")
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True

# ── TensorFlow / Keras ───────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'data_dir':       'data',
    'results_dir':    'results',
    'image_size':     224,
    'clahe_clip':     2.0,
    'clahe_tile':     (8, 8),
    'n_folds':        5,
    'random_state':   42,
    'svm_param_grid': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
    'ann_hidden':     (256, 128),    # Larger ANN for richer features
    'ann_max_iter':   300,
    'ann_alpha':      0.0001,
    # EfficientNet settings
    'efficientnet_epochs':       50,
    'efficientnet_finetune':     20,  # Extra epochs for fine-tuning
    'efficientnet_batch':        32,
    'efficientnet_patience':     10,
    'efficientnet_lr':           1e-3,
    'efficientnet_finetune_lr':  1e-5,
    'class_names':    ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
    'colors':         ['#7F77DD', '#1D9E75', '#D85A30', '#378ADD'],
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

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
    }

def print_fold_summary(name, scores):
    print(f"\n  {'─'*45}")
    print(f"  {name} — 5-Fold Summary:")
    for metric, vals in scores.items():
        print(f"    {metric.upper():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  {'─'*45}")

# ─────────────────────────────────────────────
#  KAGGLE SETUP
# ─────────────────────────────────────────────
def setup_kaggle():
    banner("STEP 1 — Dataset Setup")
    kaggle_dir  = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json):
        print(f"""
  Place your kaggle.json file in: {kaggle_dir}
  Then run this program again.
        """)
        os.makedirs(kaggle_dir, exist_ok=True)
        input("  Press Enter once kaggle.json is in place...")

    os.chmod(kaggle_json, 0o600)
    csv_path = os.path.join(CONFIG['data_dir'], 'train.csv')
    img_dir  = os.path.join(CONFIG['data_dir'], 'train_images')

    if os.path.exists(csv_path) and os.path.exists(img_dir):
        done(f"Dataset already present ({len(os.listdir(img_dir))} images)")
        return

    step("Downloading APTOS 2019 dataset...")
    os.system(f'kaggle competitions download -c aptos2019-blindness-detection -p {CONFIG["data_dir"]}')
    import zipfile
    zip_path = os.path.join(CONFIG['data_dir'], 'aptos2019-blindness-detection.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(CONFIG['data_dir'])
        os.remove(zip_path)
    done("Dataset ready!")

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (CONFIG['image_size'], CONFIG['image_size']),
                     interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=CONFIG['clahe_clip'],
                             tileGridSize=CONFIG['clahe_tile'])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    img = img.astype(np.float32) / 255.0
    return img

# ─────────────────────────────────────────────
#  IMPROVED FEATURE EXTRACTION
#  GLCM + LBP combined (richer feature space)
# ─────────────────────────────────────────────
def extract_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # GLCM features (16 values)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        glcm_feats.append(np.mean(graycoprops(glcm, prop)[0]))

    # LBP features (histogram of 26 bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)

    # Colour features — mean and std of each RGB channel (6 values)
    colour_feats = []
    for c in range(3):
        colour_feats.append(np.mean(image[:, :, c]))
        colour_feats.append(np.std(image[:, :, c]))

    # Combined: 4 + 26 + 6 = 36 features
    return np.concatenate([glcm_feats, lbp_hist, colour_feats])


def load_and_extract(force=False):
    banner("STEP 2 — Feature Extraction (GLCM + LBP + Colour)")

    feat_path  = os.path.join(CONFIG['results_dir'], 'X_features_improved.npy')
    label_path = os.path.join(CONFIG['results_dir'], 'y_labels.npy')

    if not force and os.path.exists(feat_path) and os.path.exists(label_path):
        step("Loading saved features...")
        X = np.load(feat_path)
        y = np.load(label_path)
        done(f"Loaded {X.shape[0]} samples, {X.shape[1]} features each")
        return X, y

    df      = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    img_dir = os.path.join(CONFIG['data_dir'], 'train_images')

    step(f"Extracting features from {len(df)} images (GLCM + LBP + Colour)...")
    print("     This takes about 15-20 minutes the first time. Please wait...")

    features_list, labels_list = [], []
    failed = 0
    start  = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        path = os.path.join(img_dir, row['id_code'] + '.png')
        img  = preprocess_image(path)
        if img is not None:
            features_list.append(extract_features(img))
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
    done(f"Extracted {X.shape[0]} samples | {X.shape[1]} features | "
         f"Failed: {failed} | Time: {time.time()-start:.0f}s")
    return X, y

# ─────────────────────────────────────────────
#  SMOTE — Fix class imbalance
# ─────────────────────────────────────────────
def apply_smote(X_train, y_train):
    step("Applying SMOTE to balance minority classes...")
    print(f"     Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smote = SMOTE(random_state=CONFIG['random_state'], k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"     After SMOTE:  {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

# ─────────────────────────────────────────────
#  SVM with SMOTE
# ─────────────────────────────────────────────
def run_svm(X, y):
    banner("STEP 3 — SVM with SMOTE")
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

        # Apply SMOTE only to training data
        Xtr_res, ytr_res = apply_smote(Xtr, ytr)

        gs = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=CONFIG['random_state']),
            CONFIG['svm_param_grid'], cv=3, scoring='f1_macro', n_jobs=-1
        )
        gs.fit(Xtr_res, ytr_res)
        clf = gs.best_estimator_

        y_pred = clf.predict(Xva)
        y_prob = clf.predict_proba(Xva)
        m = compute_metrics(yva, y_pred, y_prob)
        for k, v in m.items():
            all_scores[k].append(v)

        if m['f1'] > best_f1:
            best_f1, best_model = m['f1'], clf

        print(f"     Best params: C={gs.best_params_['C']}, γ={gs.best_params_['gamma']}")
        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("SVM + SMOTE", all_scores)
    return all_scores, best_model, scaler

# ─────────────────────────────────────────────
#  ANN with SMOTE
# ─────────────────────────────────────────────
def run_ann(X, y, scaler):
    banner("STEP 4 — ANN with SMOTE")
    skf  = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                           random_state=CONFIG['random_state'])
    X_sc = scaler.transform(X)
    all_scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
    best_model, best_f1 = None, 0

    for fold, (tr, va) in enumerate(skf.split(X_sc, y)):
        step(f"ANN Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_sc[tr], X_sc[va]
        ytr, yva = y[tr],    y[va]

        Xtr_res, ytr_res = apply_smote(Xtr, ytr)

        ann = MLPClassifier(
            hidden_layer_sizes=CONFIG['ann_hidden'],
            activation='relu', solver='adam',
            alpha=CONFIG['ann_alpha'], batch_size=32,
            max_iter=CONFIG['ann_max_iter'],
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=CONFIG['random_state']
        )
        ann.fit(Xtr_res, ytr_res)

        y_pred = ann.predict(Xva)
        y_prob = ann.predict_proba(Xva)
        m = compute_metrics(yva, y_pred, y_prob)
        for k, v in m.items():
            all_scores[k].append(v)

        if m['f1'] > best_f1:
            best_f1, best_model = m['f1'], ann

        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("ANN + SMOTE", all_scores)
    return all_scores, best_model

# ─────────────────────────────────────────────
#  FUZZY FUSION with SMOTE
# ─────────────────────────────────────────────
def run_fusion(X, y, scaler):
    banner("STEP 5 — Fuzzy Fusion with SMOTE")
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

        Xtr_res, ytr_res = apply_smote(Xtr, ytr)

        # Train SVM
        gs = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=CONFIG['random_state']),
            CONFIG['svm_param_grid'], cv=3, scoring='f1_macro', n_jobs=-1
        )
        gs.fit(Xtr_res, ytr_res)
        svm_clf = gs.best_estimator_

        # Train ANN
        ann = MLPClassifier(
            hidden_layer_sizes=CONFIG['ann_hidden'],
            activation='relu', solver='adam',
            alpha=CONFIG['ann_alpha'], max_iter=CONFIG['ann_max_iter'],
            early_stopping=True, random_state=CONFIG['random_state']
        )
        ann.fit(Xtr_res, ytr_res)

        # Weights from training F1
        w1 = f1_score(ytr_res, svm_clf.predict(Xtr_res), average='macro', zero_division=0)
        w2 = f1_score(ytr_res, ann.predict(Xtr_res),     average='macro', zero_division=0)
        total = w1 + w2 + 1e-9
        w1, w2 = w1/total, w2/total

        p1 = svm_clf.predict_proba(Xva)
        p2 = ann.predict_proba(Xva)
        fused = (w1 * p1) + (w2 * p2)
        y_pred = np.argmax(fused, axis=1)

        m = compute_metrics(yva, y_pred, fused)
        for k, v in m.items():
            all_scores[k].append(v)

        last_preds = y_pred
        last_true  = yva
        last_probs = fused

        print(f"     w_SVM={w1:.3f} w_ANN={w2:.3f} | "
              f"Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

    print_fold_summary("Fusion + SMOTE", all_scores)
    return all_scores, last_true, last_preds, last_probs

# ─────────────────────────────────────────────
#  EFFICIENTNET with TRANSFER LEARNING
# ─────────────────────────────────────────────
def build_efficientnet():
    """
    EfficientNetB0 with ImageNet weights.
    Strategy:
      Phase 1 — Freeze base, train top layers (fast)
      Phase 2 — Unfreeze top 30 layers, fine-tune (slow but accurate)
    """
    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(CONFIG['image_size'], CONFIG['image_size'], 3)
    )
    # Freeze all base layers initially
    base.trainable = False

    inputs = tf.keras.Input(shape=(CONFIG['image_size'], CONFIG['image_size'], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['efficientnet_lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base


def run_efficientnet(df_path, img_dir):
    banner("STEP 6 — EfficientNetB0 with Transfer Learning")

    step("Loading images for EfficientNet...")
    df = pd.read_csv(df_path)
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

    # Class weights to handle imbalance (alternative/complement to SMOTE)
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_img), y=y_img)
    class_weights = dict(enumerate(cw))
    print(f"     Class weights: { {k: round(v,2) for k,v in class_weights.items()} }")

    skf        = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True,
                                 random_state=CONFIG['random_state'])
    all_scores = {m: [] for m in ['accuracy','precision','recall','f1','auc']}

    for fold, (tr, va) in enumerate(skf.split(X_img, y_img)):
        step(f"EfficientNet Fold {fold+1}/{CONFIG['n_folds']}...")
        Xtr, Xva = X_img[tr], X_img[va]
        ytr, yva = y_img[tr], y_img[va]

        model, base = build_efficientnet()

        es  = EarlyStopping(patience=CONFIG['efficientnet_patience'],
                            restore_best_weights=True, verbose=0)
        rlr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=0)

        # Phase 1: Train top layers only
        print(f"     Phase 1: Training top layers ({CONFIG['efficientnet_epochs']} epochs max)...")
        model.fit(
            Xtr, ytr,
            epochs=CONFIG['efficientnet_epochs'],
            batch_size=CONFIG['efficientnet_batch'],
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[es, rlr],
            verbose=0
        )

        # Phase 2: Unfreeze top 30 layers and fine-tune
        print(f"     Phase 2: Fine-tuning top 30 layers ({CONFIG['efficientnet_finetune']} epochs max)...")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=CONFIG['efficientnet_finetune_lr']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        es2 = EarlyStopping(patience=CONFIG['efficientnet_patience'],
                            restore_best_weights=True, verbose=0)
        model.fit(
            Xtr, ytr,
            epochs=CONFIG['efficientnet_finetune'],
            batch_size=CONFIG['efficientnet_batch'],
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[es2],
            verbose=0
        )

        y_prob = model.predict(Xva, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        m = compute_metrics(yva, y_pred, y_prob)
        for k, v in m.items():
            all_scores[k].append(v)

        print(f"     Acc={m['accuracy']:.4f} | F1={m['f1']:.4f} | AUC={m['auc']:.4f}")

        # Save confusion matrix for last fold
        if fold == CONFIG['n_folds'] - 1:
            cm = confusion_matrix(yva, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=CONFIG['class_names'],
                        yticklabels=CONFIG['class_names'], linewidths=0.5)
            plt.title('Confusion Matrix — EfficientNetB0 (Transfer Learning)',
                      fontsize=13, fontweight='bold')
            plt.ylabel('True Label', fontsize=11)
            plt.xlabel('Predicted Label', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['results_dir'],
                        'efficientnet_confusion_matrix.png'), dpi=150)
            plt.close()

        tf.keras.backend.clear_session()

    print_fold_summary("EfficientNetB0 + Transfer Learning", all_scores)
    return all_scores

# ─────────────────────────────────────────────
#  RESULTS TABLE
# ─────────────────────────────────────────────
def print_results_table(all_results):
    banner("FINAL RESULTS TABLE")
    metrics = ['accuracy','precision','recall','f1','auc']
    names   = list(all_results.keys())
    header  = f"  {'Metric':<12}" + "".join(f"  {n:>22}" for n in names)
    print(header)
    print("  " + "─" * (12 + 24*len(names)))
    for m in metrics:
        row = f"  {m.upper():<12}"
        for name in names:
            mean = np.mean(all_results[name][m])
            std  = np.std(all_results[name][m])
            row += f"  {mean:.4f} ± {std:.4f}      "
        print(row)

    # Wilcoxon: EfficientNet vs Fusion
    if 'EfficientNet' in all_results and 'Fusion' in all_results:
        print("\n  Wilcoxon Test (EfficientNet vs Fusion):")
        for m in metrics:
            try:
                stat, p = wilcoxon(all_results['EfficientNet'][m],
                                   all_results['Fusion'][m])
                sig = "SIGNIFICANT ✓" if p < 0.05 else "not significant"
                print(f"    {m.upper():<12}: p={p:.4f} → {sig}")
            except Exception:
                print(f"    {m.upper():<12}: insufficient data")

# ─────────────────────────────────────────────
#  SAVE CSV
# ─────────────────────────────────────────────
def save_results_csv(all_results):
    metrics = ['accuracy','precision','recall','f1','auc']
    rows = []
    for name, scores in all_results.items():
        row = {'Model': name}
        for m in metrics:
            row[f'{m}_mean'] = round(np.mean(scores[m]), 4)
            row[f'{m}_std']  = round(np.std(scores[m]),  4)
        rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(CONFIG['results_dir'], 'results_improved.csv')
    df.to_csv(path, index=False)
    done(f"Results saved to {path}")
    print(df.to_string(index=False))
    return df

# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def plot_comparison(all_results):
    banner("Generating Charts")
    metrics     = ['accuracy','precision','recall','f1','auc']
    model_names = list(all_results.keys())
    colors      = ['#7F77DD','#1D9E75','#D85A30','#378ADD','#EF9F27'][:len(model_names)]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Improved Model Performance — EfficientNet + SMOTE\n5-Fold Cross-Validation on APTOS 2019',
                 fontsize=13, fontweight='bold', y=1.02)

    for i, metric in enumerate(metrics):
        means = [np.mean(all_results[m][metric]) for m in model_names]
        stds  = [np.std(all_results[m][metric])  for m in model_names]
        axes[i].bar(model_names, means, yerr=stds, color=colors,
                    capsize=5, edgecolor='white')
        axes[i].set_title(metric.upper(), fontsize=11, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=20)
        for j, (m2, s) in enumerate(zip(means, stds)):
            axes[i].text(j, m2 + s + 0.01, f'{m2:.3f}',
                         ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'comparison_improved.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved: {path}")

def plot_per_fold(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Fold Performance — Improved Pipeline', fontsize=13, fontweight='bold')
    model_names = list(all_results.keys())
    colors      = ['#7F77DD','#1D9E75','#D85A30','#378ADD','#EF9F27'][:len(model_names)]
    folds       = list(range(1, CONFIG['n_folds']+1))
    for ax, metric in zip(axes.flat, ['accuracy','f1','recall','auc']):
        for name, col in zip(model_names, colors):
            ax.plot(folds, all_results[name][metric], marker='o',
                    label=name, color=col, linewidth=2)
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'per_fold_improved.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    done(f"Saved: {path}")

def plot_glcm_features(X, y):
    feature_names = ['Contrast','Correlation','Energy','Homogeneity']
    colors_feat   = ['#7F77DD','#1D9E75','#D85A30','#378ADD','#EF9F27']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('GLCM Feature Distributions by DR Grade', fontsize=13, fontweight='bold')
    for i, feat in enumerate(feature_names):
        for c in range(5):
            vals = X[y==c, i]
            axes[i].hist(vals, alpha=0.6, label=CONFIG['class_names'][c],
                         color=colors_feat[c], bins=25)
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xlabel('Feature Value')
        axes[i].set_ylabel('Frequency')
        if i == 0:
            axes[i].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(CONFIG['results_dir'], 'glcm_distributions_improved.png')
    plt.savefig(path, dpi=150)
    plt.close()
    done(f"Saved: {path}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   IMPROVED DR Detection — EfficientNet + SMOTE           ║
║   Chandra Kiran Malkapuram | U2904743 | UEL              ║
╠══════════════════════════════════════════════════════════╣
║  Improvements over v1:                                   ║
║  • EfficientNetB0 with ImageNet transfer learning        ║
║  • SMOTE oversampling for minority DR classes            ║
║  • Extended features: GLCM + LBP + Colour (36 dims)     ║
║  • Two-phase fine-tuning for EfficientNet                ║
║  • Class weights for CNN training                        ║
╚══════════════════════════════════════════════════════════╝

  Select an option:
  [1]  Full pipeline (EfficientNet + Fusion + SVM + ANN) — ~2-3 hours
  [2]  Fusion + SVM + ANN only (no EfficientNet) — ~45 min
  [3]  EfficientNet only — ~1-2 hours
  [4]  Exit
    """)

    choice = input("  Enter choice (1/2/3/4): ").strip()
    if choice == '4':
        sys.exit(0)

    setup_kaggle()
    df_path = os.path.join(CONFIG['data_dir'], 'train.csv')
    img_dir = os.path.join(CONFIG['data_dir'], 'train_images')
    all_results = {}

    if choice in ['1', '2']:
        # Feature extraction + classical ML
        X, y = load_and_extract()
        plot_glcm_features(X[:, :4], y)  # First 4 are GLCM

        svm_scores, best_svm, scaler = run_svm(X, y)
        all_results['SVM+SMOTE'] = svm_scores

        ann_scores, best_ann = run_ann(X, y, scaler)
        all_results['ANN+SMOTE'] = ann_scores

        fusion_scores, f_true, f_pred, f_prob = run_fusion(X, y, scaler)
        all_results['Fusion+SMOTE'] = fusion_scores

        # Confusion matrix for fusion
        cm = confusion_matrix(f_true, f_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CONFIG['class_names'],
                    yticklabels=CONFIG['class_names'], linewidths=0.5)
        plt.title('Confusion Matrix — Fusion Framework + SMOTE',
                  fontsize=13, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['results_dir'],
                    'fusion_confusion_matrix_improved.png'), dpi=150)
        plt.close()

    if choice in ['1', '3']:
        # EfficientNet
        eff_scores = run_efficientnet(df_path, img_dir)
        all_results['EfficientNet'] = eff_scores

    # Results
    print_results_table(all_results)
    save_results_csv(all_results)
    plot_comparison(all_results)
    plot_per_fold(all_results)

    banner("ALL DONE!")
    print(f"""
  Results saved to the 'results/' folder:

    results_improved.csv               — All metrics table
    comparison_improved.png            — Bar chart comparison
    per_fold_improved.png              — Per-fold line plots
    glcm_distributions_improved.png    — Feature distributions
    fusion_confusion_matrix_improved.png
    efficientnet_confusion_matrix.png  — EfficientNet confusion matrix

  Upload these results and come back to update Chapter 4!
    """)

if __name__ == '__main__':
    main()
