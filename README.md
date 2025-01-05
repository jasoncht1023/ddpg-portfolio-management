# Setup

## 1. Set up and activate the Python virtual environment
Python 3.11 is required (Recommended version: 3.11.5)
```bash
    pip install virtualenv
    virtualenv -p python3.11.5 .venv
    .venv/Scripts/activate
```
## 2. Install the dependencies
```bash
    pip install -r ./requirements.txt
```
## 3. Install PyTorch (Version 2.5.1)
Example installation: 
```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Or look for customized installation command in https://pytorch.org/

# Updating the dependencies
Update the dependency list after you installed new packages through pip using the `deps_update.bat` script \
While in the virtual environment, run:
```bash
    ./deps_update.bat
```

# Evaluation
Total return and Sharpe Ratio over epoch will be generated in `/evaluation` after the training is completed