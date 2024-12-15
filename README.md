# MLFMM Repository

This repository contains exercises and implementations for the Machine Learning Fundamentals and Methods course.

## Large Data Files Setup

Some large data files are not included in this repository due to GitHub's file size constraints. Follow the instructions below to set up these files:

### 1. CIFAR-10 Dataset
Location: `UE/10/data/cifar-10-python.tar.gz`

To set up the CIFAR-10 dataset:
1. Download the Python version of CIFAR-10 from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. Create the directory if it doesn't exist: `mkdir -p UE/10/data`
3. Place the downloaded file in the `UE/10/data/` directory
4. Verify the file is named exactly `cifar-10-python.tar.gz`

### 2. Training Data File
Location: `UE/07/Pset07_files/X.npy`

This is a large numpy array file used for training. To obtain this file:
1. Download it from [insert your preferred hosting location - e.g., course website, shared drive]
2. Create the directory if it doesn't exist: `mkdir -p UE/07/Pset07_files`
3. Place the file in the `UE/07/Pset07_files/` directory
4. Verify the file is named exactly `X.npy`

## Repository Structure

- `UE/`: Contains all exercise materials and implementations
- `03/`, `04/`, `05/`, `07/`, `10/`: Individual exercise folders
- Each folder contains its specific implementation files and datasets

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/YannickHradetzky/MLFMM.git
```
2. Follow the "Large Data Files Setup" instructions above
3. Install required dependencies (if any)

## Note About Large Files

Files larger than 100MB are not tracked in this repository. If you need to add any large files in the future, please:
1. Check if they can be regenerated from code
2. If not, add them to .gitignore and update this README with download instructions
