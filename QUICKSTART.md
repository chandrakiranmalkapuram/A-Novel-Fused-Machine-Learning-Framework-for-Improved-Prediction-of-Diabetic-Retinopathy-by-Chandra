# 🚀 Quick Start Guide

## Installation (2 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

**That's it!** Your browser will open to `http://localhost:8501`

---

## How to Use (3 simple steps)

### Step 1: Upload Dataset 📤
- CSV file with columns: `id_code`, `diagnosis`
- Image files (PNG/JPG) matching the IDs
- Click "Load Dataset for Processing"

### Step 2: Train Models 🔬
- Select models: SVM ✓ ANN ✓ Fusion ✓ CNN ☐
- Click "Start Training"
- Wait 5-30 minutes

### Step 3: View Results 📈
- Summary statistics table
- 4 visualization tabs
- Download as CSV/PNG

---

## File Guide

| File | Purpose |
|------|---------|
| `app.py` | Main web application (RUN THIS) |
| `processing.py` | ML functions & models |
| `requirements.txt` | Python dependencies |
| `WEB_APP_README.md` | Detailed documentation |
| `SETUP_GUIDE.md` | Installation guide |
| `CONVERSION_SUMMARY.md` | What was created & why |
| `uploads/` | Temporary file storage |
| `outputs/` | Results folder |

---

## CSV Format

```csv
id_code,diagnosis
image_001,0
image_002,1
image_003,2
image_004,3
image_005,4
```

- `id_code`: Must match image filename (without extension)
- `diagnosis`: DR grade (0-4)

---

## Image Requirements

- Format: PNG or JPG
- Naming: `image_001.png`, `image_002.jpg`, etc.
- Size: Any (will be resized to 224×224)
- Color: RGB fundus images

---

## Models Explained

| Model | Time | Speed |
|-------|------|-------|
| SVM | 5-10 min | ⚡⚡⚡ Fast |
| ANN | 5-10 min | ⚡⚡⚡ Fast |
| Fusion | 10-15 min | ⚡⚡⚡ Fast |
| CNN | 30+ min | 🐢 Very slow (use GPU) |

**Recommendation:** Start with SVM + ANN + Fusion for best results

---

## Troubleshooting

### Images not loading?
- Check filenames match CSV id_code
- Use extensions: `.png` or `.jpg`
- No spaces in filenames

### Out of memory?
- Use fewer images
- Don't train CNN
- Close other programs

### App very slow?
- Skip CNN option
- Use smaller dataset
- Check CPU/RAM usage

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

---

## What's New vs Original

```
Original main.py (1000 lines)
         ↓
    Split into:

app.py (450 lines)
  └─ Web interface
     Streamlit UI
     File upload/download
     Result display

processing.py (550 lines)
  └─ ML Core
     All functions
     Reusable & clean
```

---

## Downloads

When you click download on the Results page, you get:

- `results_summary.csv` - All metrics table
- `comparison_chart.png` - Model comparison chart
- `confusion_matrix_fusion.png` - Fusion confusion matrix
- `glcm_distributions.png` - Feature distributions
- `per_fold_performance.png` - Fold-by-fold metrics

---

## For Your Dissertation

Perfect for creating publication-ready figures:

1. Run training in web app
2. Download PNG charts
3. Embed in Word/LaTeX
4. Download CSV for results table
5. Insert into Chapter 4

---

## Complete Documentation

For more details, see:

- **`SETUP_GUIDE.md`** - Step-by-step installation
- **`WEB_APP_README.md`** - Full user guide (500+ lines)
- **`CONVERSION_SUMMARY.md`** - Technical details & code mapping
- **`processing.py`** - Function comments & details
- **`app.py`** - UI code with inline comments

---

## Quick Test

Try with minimal data:

1. Create `test.csv` with 5 rows
2. Add 5 sample images (any 224×224 images work)
3. Upload through web app
4. Select only SVM
5. Should complete in 2-3 minutes ✓

---

## Support

**Error?** Check the relevant guide:
- Installation issue → `SETUP_GUIDE.md`
- Data format issue → `WEB_APP_README.md`
- Code question → `CONVERSION_SUMMARY.md`
- Function details → `processing.py` comments

**Still stuck?** 
- Google the error message
- Ask your supervisor
- Review Streamlit docs: https://docs.streamlit.io

---

## Summary

✅ Installation: `pip install -r requirements.txt`  
✅ Run: `streamlit run app.py`  
✅ Upload: CSV + images  
✅ Train: Select models  
✅ Results: View & download  

**You're ready to go!** 🎉

---

*Created for your MSc Dissertation | Dr. Fidal Bashir, UEL*
