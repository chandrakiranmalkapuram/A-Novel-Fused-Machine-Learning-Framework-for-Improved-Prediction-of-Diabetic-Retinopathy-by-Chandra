"""
EfficientNetB0 Transfer Learning — Mac Version
Chandra Kiran Malkapuram | U2904743 | UEL

Run with:  python3.11 efficientnet_mac.py
"""

# Force TensorFlow to use CPU only — fixes Mac GPU collapse issue
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

print("="*60)
print("  EfficientNetB0 — DR Detection (Mac CPU Version)")
print("  Chandra Kiran Malkapuram | U2904743 | UEL")
print("="*60)
print(f"\nTensorFlow version: {tf.__version__}")
print("Running on: CPU (stable Mac mode)")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMAGE_SIZE  = 224
N_FOLDS     = 5
BATCH_SIZE  = 16    # Smaller batch = less memory needed on Mac
EPOCHS_P1   = 15
EPOCHS_P2   = 15
PATIENCE    = 5
LR_P1       = 1e-3
LR_P2       = 1e-5
IMG_DIR     = 'data/train_images/'
CSV_PATH    = 'data/train.csv'
RESULTS_DIR = 'results/'
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CHECK DATA EXISTS
# ─────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"\nERROR: Cannot find {CSV_PATH}")
    print("Please make sure your data folder contains:")
    print("  data/train.csv")
    print("  data/train_images/")
    print("\nIf you have already downloaded the dataset, make sure")
    print("you are running this script from inside the dr_project folder.")
    exit(1)

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                     interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[:,:,1] = clahe.apply(img[:,:,1])
    return img.astype(np.float32) / 255.0

# ─────────────────────────────────────────────
# LOAD IMAGES
# ─────────────────────────────────────────────
feat_path  = os.path.join(RESULTS_DIR, 'X_images.npy')
label_path = os.path.join(RESULTS_DIR, 'y_labels_eff.npy')

if os.path.exists(feat_path) and os.path.exists(label_path):
    print("\nLoading saved images from disk...")
    X = np.load(feat_path)
    y = np.load(label_path)
    print(f"Loaded {X.shape[0]} images!")
else:
    print("\nLoading and preprocessing all images...")
    print("This takes about 10-15 minutes the first time...")
    df = pd.read_csv(CSV_PATH)
    imgs, labels = [], []
    failed = 0
    start  = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        img = preprocess_image(IMG_DIR + row['id_code'] + '.png')
        if img is not None:
            imgs.append(img)
            labels.append(row['diagnosis'])
        else:
            failed += 1
        if (i+1) % 300 == 0:
            elapsed = time.time() - start
            eta     = (elapsed/(i+1)) * (len(df)-i-1)
            print(f"  {i+1}/{len(df)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    X = np.array(imgs,   dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    np.save(feat_path,  X)
    np.save(label_path, y)
    print(f"Saved images to disk for next time!")

print(f"\nDataset: {X.shape[0]} images | Shape: {X.shape}")
print("Class distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(y == i)
    pct   = count / len(y) * 100
    print(f"  Grade {i} ({name}): {count} images ({pct:.1f}%)")

# ─────────────────────────────────────────────
# METRICS
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
# BUILD EFFICIENTNET
# ─────────────────────────────────────────────
def build_efficientnet():
    # Load EfficientNetB0 pre-trained on ImageNet
    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base.trainable = False  # Freeze base for Phase 1

    # Add classification head for 5 DR grades
    inputs  = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
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
        optimizer=Adam(LR_P1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base

# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(cw))
print("\nClass weights (higher = rarer class needs more attention):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  Grade {i} ({name}): {class_weights[i]:.3f}")

# ─────────────────────────────────────────────
# 5-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Starting 5-Fold Cross-Validation...")
print(f"This will take approximately 2-3 hours on Mac CPU.")
print(f"Leave this terminal open and let it run.")
print("="*60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
all_scores  = {m: [] for m in ['accuracy','precision','recall','f1','auc']}
last_y_true = None
last_y_pred = None
last_y_prob = None
total_start = time.time()

for fold, (tr, va) in enumerate(skf.split(X, y)):
    print(f"\n{'─'*50}")
    print(f"FOLD {fold+1} of {N_FOLDS}")
    print(f"{'─'*50}")
    fold_start = time.time()

    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]
    print(f"Training: {len(Xtr)} images | Validation: {len(Xva)} images")

    model, base = build_efficientnet()

    # ── Phase 1: Train top layers only ──────────────────────
    print(f"\nPhase 1: Training classification head...")
    es1 = EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                        restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=3, verbose=1)
    model.fit(
        Xtr, ytr,
        epochs=EPOCHS_P1,
        batch_size=BATCH_SIZE,
        validation_data=(Xva, yva),
        class_weight=class_weights,
        callbacks=[es1, rlr],
        verbose=1
    )
    best_p1 = max(model.history.history.get('val_accuracy', [0]))
    print(f"Phase 1 best val accuracy: {best_p1:.4f}")

    # ── Phase 2: Fine-tune top 30 layers ────────────────────
    print(f"\nPhase 2: Fine-tuning top 30 EfficientNet layers...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(LR_P2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    es2 = EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                        restore_best_weights=True, verbose=1)
    model.fit(
        Xtr, ytr,
        epochs=EPOCHS_P2,
        batch_size=BATCH_SIZE,
        validation_data=(Xva, yva),
        class_weight=class_weights,
        callbacks=[es2],
        verbose=1
    )

    # ── Evaluate ─────────────────────────────────────────────
    y_prob = model.predict(Xva, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    m = compute_metrics(yva, y_pred, y_prob)

    for k, v in m.items():
        all_scores[k].append(v)

    fold_mins = (time.time() - fold_start) / 60
    print(f"\nFold {fold+1} Results:")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1-Score : {m['f1']:.4f}")
    print(f"  AUC      : {m['auc']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  Fold time: {fold_mins:.1f} minutes")

    # Save last fold for confusion matrix
    if fold == N_FOLDS - 1:
        last_y_true = yva
        last_y_pred = y_pred
        last_y_prob = y_prob

    # Partial results saved after every fold
    # so you don't lose everything if it crashes
    partial = pd.DataFrame([{
        'Fold': fold+1,
        **{k: round(v, 4) for k, v in m.items()}
    }])
    partial_path = os.path.join(RESULTS_DIR, f'fold_{fold+1}_result.csv')
    partial.to_csv(partial_path, index=False)
    print(f"  Saved fold result to {partial_path}")

    del model
    tf.keras.backend.clear_session()

# ─────────────────────────────────────────────
# FINAL RESULTS
# ─────────────────────────────────────────────
total_mins = (time.time() - total_start) / 60
print("\n" + "="*60)
print("FINAL RESULTS — EfficientNetB0 Transfer Learning")
print("="*60)
for metric, vals in all_scores.items():
    print(f"  {metric.upper():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print(f"\nTotal training time: {total_mins:.1f} minutes")

# Save CSV
rows = {'Model': 'EfficientNet'}
for m in all_scores:
    rows[f'{m}_mean'] = round(np.mean(all_scores[m]), 4)
    rows[f'{m}_std']  = round(np.std(all_scores[m]),  4)
pd.DataFrame([rows]).to_csv(
    os.path.join(RESULTS_DIR, 'efficientnet_results.csv'), index=False
)
print("\nResults saved to results/efficientnet_results.csv")

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(last_y_true, last_y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5, annot_kws={'size': 13})
plt.title('Confusion Matrix — EfficientNetB0 (Transfer Learning)\nAPTOS 2019 Dataset',
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel('True Label',      fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'efficientnet_confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
print("Confusion matrix saved!")

# Per-class accuracy
print("\nPer-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    correct = cm[i, i]
    total   = cm[i, :].sum()
    pct     = correct/total*100 if total > 0 else 0
    print(f"  {name:<12}: {correct}/{total} correct ({pct:.1f}%)")

# ─────────────────────────────────────────────
# PER-FOLD PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
folds = list(range(1, N_FOLDS+1))

axes[0].plot(folds, all_scores['accuracy'], marker='o',
             color='#1D9E75', linewidth=2.5, markersize=8, label='Accuracy')
axes[0].plot(folds, all_scores['f1'],       marker='s',
             color='#534AB7', linewidth=2.5, markersize=8, label='F1-Score')
axes[0].plot(folds, all_scores['recall'],   marker='^',
             color='#D85A30', linewidth=2.5, markersize=8, label='Recall')
axes[0].set_title('EfficientNetB0 — Per-Fold Performance',
                  fontweight='bold', fontsize=12)
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(folds, all_scores['auc'], marker='o',
             color='#BA7517', linewidth=2.5, markersize=8)
axes[1].set_title('EfficientNetB0 — AUC Per Fold',
                  fontweight='bold', fontsize=12)
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('AUC')
axes[1].set_ylim(0, 1.05)
axes[1].grid(alpha=0.3)

plt.suptitle(
    f'EfficientNetB0 Transfer Learning | '
    f'Mean Accuracy: {np.mean(all_scores["accuracy"]):.4f} | '
    f'Mean F1: {np.mean(all_scores["f1"]):.4f}',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'efficientnet_per_fold.png'),
            dpi=150, bbox_inches='tight')
print("Per-fold plot saved!")

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)
print("\nFiles saved in your results/ folder:")
print("  efficientnet_results.csv")
print("  efficientnet_confusion_matrix.png")
print("  efficientnet_per_fold.png")
print("\nSuccess!")
