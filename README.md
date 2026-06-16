# Diabetic Retinopathy Detection — Setup Guide
### Chandra Kiran Malkapuram | U2904743 | University of East London

---

## What this program does

This program runs your full dissertation experiment automatically:
- Downloads the APTOS 2019 eye dataset from Kaggle
- Extracts GLCM texture features from all 3,662 images
- Trains SVM, ANN, and Fuzzy Fusion models
- Trains a CNN baseline for comparison
- Saves all results, charts and tables ready for Chapter 4

---

## Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. During installation, tick the box that says **"Add Python to PATH"**
4. Click Install

---

## Step 2 — Install VS Code

1. Go to https://code.visualstudio.com/
2. Download and install VS Code
3. Open VS Code
4. Open the folder containing this project:
   File → Open Folder → select the `dr_project` folder

---

## Step 3 — Open the Terminal in VS Code

- Go to **View → Terminal**
- A black terminal window will appear at the bottom

---

## Step 4 — Install the required libraries

Type this in the terminal and press Enter:

```
pip install -r requirements.txt
```

Wait for everything to install (5-10 minutes).

---

## Step 5 — Get your Kaggle API key

1. Go to https://www.kaggle.com and sign in (or create a free account)
2. Click your profile picture (top right) → **Settings**
3. Scroll to the **API** section
4. Click **"Create New Token"**
5. A file called `kaggle.json` will download to your computer
6. The program will tell you exactly where to put it when you run it

---

## Step 6 — Run the program

Type this in the terminal and press Enter:

```
python main.py
```

You will see a menu:
- Choose **[1]** for the full pipeline (SVM + ANN + Fusion + CNN) — takes 1-2 hours
- Choose **[2]** for SVM + ANN + Fusion only (no CNN) — takes ~30 minutes
- Choose **[3]** to just view saved results

---

## What you get in the results/ folder

| File | What it is |
|---|---|
| `results_summary.csv` | Table of all metrics for Chapter 4 |
| `comparison_chart.png` | Bar chart comparing all models |
| `fusion_confusion_matrix.png` | Confusion matrix figure |
| `glcm_distributions.png` | GLCM feature distribution plots |
| `per_fold_performance.png` | Per-fold line plots |

---

## If something goes wrong

- If you see a red error message, copy it and send it to your assistant
- If the program stops halfway, just run it again — it saves progress automatically
- If Kaggle download fails, make sure you accepted the competition rules at:
  https://www.kaggle.com/c/aptos2019-blindness-detection

---

## Important notes

- The first run takes 1-2 hours (downloading + training)
- After the first run, results are saved so re-running is instant
- Keep VS Code open while it runs — do not close the terminal
- Your computer needs to be plugged in (not on battery) for best performance
