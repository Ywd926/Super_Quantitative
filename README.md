Digital PCR & Digital ELISA Image Analyzer
=========
A Python-based GUI application for automated analysis of digital PCR (dPCR) and digital ELISA (dELISA) images. This tool streamlines the process of microreactor identification, classification, and data export for quantitative biological assays.

Features
---------
·Image Import​​: Supports common image formats for dPCR/dELISA experiments

·Microreactor Detection​​: Automated identification of reaction chambers/wells

·Interactive Labeling​​: Manual annotation interface for training and validation

·Visualization​​: Real-time display of classification results with customizable overlays

·​​Data Export​​: Save analysis results in multiple formats for downstream processing

Quick Start
---------

Installation

1. Clone the repository
```Bash
git clone https://github.com/Ywd926/Super_Quantitative.git
cd Super_Quantitative
```
2. Clone the repository
```Bash
pip install -r requirements.txt
```

Usage

Launch the application:
```Bash
python main.py
```

Workflow:

​​1.Import Images​​: Load your dPCR/dELISA images through the File menu

​​2.Detect Microreactors​​: Automatic segmentation of reaction units

​​3.Manual Annotation​​: Label subsets of reactors for training/validation

​​4.Review Results​​: Visualize classification predictions with interactive overlay

​​5.Export Data​​: Save quantitative results in CSV/Excel format

Requirements
---------
See requirements.txtfor complete dependency list.

Graphical User Interface
---------
![](https://github.com/Ywd926/Super_Quantitative/raw/main/GUI.png)


Example
---------
![](https://github.com/Ywd926/Super_Quantitative/raw/main/test.svg)

