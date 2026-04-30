# PRC_test Project - Environment Setup Guide

## Quick Start (開始)

### Method 1: Using the Setup Script (推薦方法)

```bash
# Step 1: Navigate to the project directory
cd /Users/yumengzheng/Desktop/program/PRC_test

# Step 2: Run the setup script
python setup_environment.py
```

This script will:
- ✓ Check all required packages
- ✓ Display installation status
- ✓ Automatically install missing packages
- ✓ Verify the complete environment

### Method 2: Using requirements.txt

```bash
# Install all packages at once
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, matplotlib, sklearn; print('✓ All packages installed successfully')"
```

### Method 3: Manual Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Required Packages (必要のパッケージ)

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | >=1.20.0 | Numerical computing and array operations |
| **pandas** | >=1.3.0 | Data manipulation and CSV file handling |
| **matplotlib** | >=3.4.0 | Data visualization and plotting |
| **scikit-learn** | >=0.24.0 | Ridge regression (RidgeCV) for ML |

---

## Project Files (ファイル内容リスト)

- **setup_environment.py** - Environment preparation and package verification
- **requirements.txt** - List of all required packages
- **auto_STM_batch.py** - Batch processing for multiple CSV files
- **STM.py** - Single file STM analysis
- **STM_full.py** - Full STM analysis with detailed output

---

## Typical Workflow 

```bash
# 1. First time setup
python setup_environment.py

# 2. Edit configuration in the script
# Set INPUT_DIR and OUTPUT_DIR paths

# 3. Run analysis
python auto_STM_batch.py      # For batch processing
# or
python STM.py                 # For single file
# or
python STM_full.py            # For detailed analysis
```

---

## Troubleshooting (故障排除)

### Issue: "ModuleNotFoundError: No module named 'xxx'"
**Solution:** Run `python setup_environment.py` again or manually install:
```bash
pip install <package_name>
```

### Issue: Permission denied when installing
**Solution:** Use `pip install --user` instead:
```bash
python setup_environment.py  # This handles it automatically
```

### Issue: Conda environment
**Solution:** If using conda:
```bash
conda activate base
python setup_environment.py
```

---

## Verifying Setup (設置確認)

To verify all packages are correctly installed:

```bash
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
print('✓ NumPy:', np.__version__)
print('✓ Pandas:', pd.__version__)
print('✓ Matplotlib:', plt.__version__)
print('✓ Scikit-learn installed')
print('All packages ready!')
"
```

Or simply run:
```bash
python setup_environment.py
```

---

## Notes (注意事項)

- The setup script uses `pip` for package management
- Internet connection required for first-time installation
- Python 3.7+ is recommended
- The script will not modify existing package versions unless necessary

---

*Last updated: 2026-04-30*
