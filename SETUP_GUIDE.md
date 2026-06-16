# Installation & Setup Guide

Quick start guide to get your Diabetic Retinopathy Detection web application running.

## ✅ Prerequisites

- Python 3.8+ installed
- pip (Python package manager)
- Terminal/Command Prompt access
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Installation Steps

### Step 1: Install Dependencies

```bash
# Navigate to your project directory
cd /path/to/dr_project

# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- Streamlit (web framework)
- TensorFlow/Keras (for CNN and ANN)
- scikit-learn (SVM)
- OpenCV (image processing)
- Matplotlib/Seaborn (plotting)
- And 10+ other dependencies

**Installation time**: 10-30 minutes (first time)

### Step 2: Run the Application

```bash
streamlit run app.py
```

**What happens:**
1. Streamlit starts a local server
2. Your browser opens automatically to `http://localhost:8501`
3. You'll see the web app interface

### Step 3: Prepare Your Data

Before uploading, organize:

```
MyDataset/
├── data.csv              (CSV file with id_code and diagnosis)
└── images/               (folder with retinal images)
    ├── image_001.png
    ├── image_002.jpg
    ├── image_003.png
    └── ... (more images)
```

**CSV Format:**
```csv
id_code,diagnosis
image_001,0
image_002,1
image_003,2
image_004,3
image_005,4
```

### Step 4: Use the Web App

**Page 1: Upload Dataset** 📤
1. Click "Upload CSV file" → select your data.csv
2. Click "Upload images" → select all retinal images
3. Click "Load Dataset for Processing"
4. Wait for feature extraction to complete

**Page 2: Processing** 🔬
1. Select which models to train:
   - ✓ SVM (recommended, fast)
   - ✓ ANN (recommended, fast)
   - ✓ Fusion (recommended, very good)
   - ☐ CNN (optional, slow)
2. Click "Start Training"
3. Wait for training to complete (5-30 minutes)

**Page 3: Results** 📈
1. View summary statistics table
2. Explore 4 visualization tabs
3. Download results as CSV or PNG

---

## 🐛 Troubleshooting

### Issue: Command not found: `streamlit`

```bash
# Make sure you're in the right directory
cd /path/to/dr_project

# Try installing streamlit specifically
pip install streamlit

# Then run again
streamlit run app.py
```

### Issue: Module not found errors

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install specific package
pip install tensorflow
pip install scikit-learn
pip install opencv-python
```

### Issue: Images not loading

**Check:**
1. Image filenames match CSV `id_code` exactly
2. File extensions are .png or .jpg
3. Images are not corrupted
4. Filenames have no spaces (use `image_001.png` not `image 001.png`)

### Issue: Out of memory error

**Solutions:**
1. Upload fewer images for testing
2. Don't check the CNN option
3. Close other applications
4. Use a machine with more RAM

### Issue: Very slow processing

**Optimize:**
1. Reduce number of images
2. Skip CNN training
3. Use a faster computer with more CPU cores
4. On Mac: Consider using `pip install tensorflow-macos` for M1/M2

---

## 📦 What's Installed

Run this command to see what was installed:

```bash
pip list | grep -E "streamlit|tensorflow|scikit|opencv|pandas|numpy"
```

You should see packages like:
- streamlit
- tensorflow
- scikit-learn
- scikit-image
- opencv-python
- pandas
- numpy
- matplotlib

---

## 🔄 File Structure

Your project should look like:

```
dr_project/
├── app.py                    ← Main web app (run this!)
├── processing.py             ← ML processing module
├── requirements.txt          ← Dependencies
├── WEB_APP_README.md         ← Detailed documentation
├── SETUP_GUIDE.md            ← This file
├── uploads/                  ← Temporary uploads (auto-created)
├── outputs/                  ← Results (auto-created)
└── data/                     ← Your dataset (create or copy here)
    ├── train.csv
    ├── train.csv
    ├── train_images/
    └── test_images/
```

---

## ⚙️ Advanced Configuration

### Modify Model Parameters

Edit `processing.py` and change the `CONFIG` dictionary:

```python
CONFIG = {
    'n_folds': 5,              # Change number of folds (3-10)
    'image_size': 224,         # Image resize dimension (128-512)
    'ann_hidden': (100,),      # ANN layers (e.g., (100, 50))
    'svm_param_grid': {...},   # SVM grid search parameters
}
```

### Use Your Own Data

1. Organize as shown above
2. Upload through web interface
3. Or place in `data/` folder and modify code

---

## 🎯 Quick Testing

To test if everything works:

```bash
# 1. Run the app
streamlit run app.py

# 2. In the browser, use SAMPLE DATA:
#    - Create sample.csv with 5-10 rows
#    - Add 5-10 sample images
#    - Click Upload → Load Dataset
#    - Select only SVM on Processing page
#    - Click Start Training (should take ~2-3 minutes)
#    - View results
```

---

## 🆘 Still Having Issues?

**Check these resources:**

1. **Streamlit Docs**: https://docs.streamlit.io/
2. **TensorFlow Docs**: https://www.tensorflow.org/
3. **scikit-learn Docs**: https://scikit-learn.org/
4. **YouTube**: Search "Streamlit tutorial" or "scikit-learn SVM"

**Common Fixes:**
```bash
# Clear cache and reinstall
pip uninstall -y streamlit tensorflow scikit-learn scikit-image
pip install -r requirements.txt

# Update pip
pip install --upgrade pip

# Use specific Python version
python3 -m pip install -r requirements.txt
```

---

## ✨ You're All Set!

Your web application is ready to use:

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

**Happy analyzing!** 🏥📊

---

**Need Help?**
- Check `WEB_APP_README.md` for detailed documentation
- Review code comments in `app.py` and `processing.py`
- Test with sample data first
- Ask your supervisor or teaching team

