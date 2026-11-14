# Adversarially Robust Breast Cancer Classification

ResNet-50 with PGD Adversarial Training for benign vs. malignant classification of breast histopathology images (BreakHis, 400x magnification), with a Streamlit web app and Grad-CAM explanations.

## Project Structure

breast_cancer_project/
├── breast_cancer_colab.py     # Training and evaluation code (Google Colab / Kaggle)
├── requirements.txt           # Required libraries
├── models/                    # Saved weights (baseline_resnet50.pth, robust_resnet50.pth)
├── data/
│   ├── raw/                   # Place the BreakHis 400X images here
│   └── processed/
└── streamlit_app/
    └── app.py                 # Streamlit web application

## Dataset

Name:    BreakHis (Breast Cancer Histopathological Database)
Source:  https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
Kaggle:  https://www.kaggle.com/datasets/ambarish/breakhis

Only the 400X magnification images are used, with all benign subtypes merged
into one class and all malignant subtypes into the other:

data/raw/400X/
  benign/       (461 images)
  malignant/    (1,232 images)

Total: 1,693 images. Corrupted files were checked for and removed before training.

## Method Summary

- Model: ResNet-50 pretrained on ImageNet, final layer replaced with Linear(2048, 2), all layers fine-tuned
- Preprocessing: resize to 224x224, ImageNet normalisation, augmentation (flips, rotation, colour jitter) on training images only
- Class imbalance: class-weighted CrossEntropyLoss
- Attacks evaluated: FGSM and PGD (Linf norm, eps = 0.01, 0.03, 0.05)
- Defense: PGD Adversarial Training (Madry et al., 2018), eps = 8/255, trained on a 50/50 mix of clean and adversarial loss with gradient clipping, initialised from the baseline weights
- Explainability: Grad-CAM

## How to Run the Training (Google Colab or Kaggle)

1. Open Google Colab or a Kaggle Notebook and set the accelerator to GPU (T4)
2. Add the BreakHis 400X dataset (upload to Drive for Colab, or "Add Data" on Kaggle)
3. Update DATA_DIR and MODEL_DIR in the config cell to match your paths
4. Run the cells in order: install, imports, config, data loading, baseline training,
   attack evaluation, adversarial training, final evaluation
5. Download baseline_resnet50.pth and robust_resnet50.pth when training finishes

Note: Colab's free GPU limit may run out mid-training. Kaggle gives a separate
weekly GPU quota and is a good fallback.

## How to Run the Streamlit App

1. Copy robust_resnet50.pth into the models/ folder
2. Install requirements:   pip install -r requirements.txt
3. Run the app:            streamlit run streamlit_app/app.py
4. Open http://localhost:8501 in your browser
5. Upload a histopathology image to get the prediction, confidence scores and Grad-CAM heatmap

## Results

### Clean test data

| Metric            | Baseline ResNet-50 | Adversarially Trained |
|-------------------|--------------------|-----------------------|
| Accuracy          | 97.43%             | 89.36%                |
| F1-Score (macro)  | 97.06%             | 87.72%                |
| AUC-ROC           | 99.57%             | 95.24%                |

### Baseline under attack

| Attack | eps = 0.01 | eps = 0.03 | eps = 0.05 |
|--------|------------|------------|------------|
| FGSM   | 68.26%     | 67.89%     | 67.71%     |
| PGD    | 62.94%     | 28.99%     | 9.54%      |

### Baseline vs. adversarially trained model (eps = 0.03)

| Metric              | Baseline | Robust | Change   |
|---------------------|----------|--------|----------|
| FGSM accuracy       | 67.89%   | 69.36% | +1.47%   |
| PGD accuracy        | 28.99%   | 69.17% | +40.18%  |
| PGD accuracy drop   | 29.91%   | 20.18% | -9.73%   |

### Key findings

- The baseline model is very accurate on clean data (97.43%) but collapses under PGD attack (9.54% at eps = 0.05).
- PGD Adversarial Training raised PGD accuracy at eps = 0.03 from 28.99% to 69.17%.
- This costs about 8% clean accuracy (97.43% to 89.36%), the usual accuracy-robustness trade-off.
- FGSM robustness improved only slightly (+1.47%).
- Grad-CAM heatmaps show the robust model focuses on stromal and cellular boundary regions.

## Limitations

- Small dataset (1,693 images) at a single magnification level (400X)
- Only white-box FGSM and PGD attacks were evaluated
- Adversarial training was limited to 20 epochs because of GPU constraints
- Not validated in a real clinical setting. Research use only, not a diagnostic tool.

## References

Madry et al. (2018). Towards deep learning models resistant to adversarial attacks. ICLR 2018.
Goodfellow et al. (2015). Explaining and harnessing adversarial examples. ICLR 2015.
He et al. (2016). Deep residual learning for image recognition. CVPR 2016.
Selvaraju et al. (2017). Grad-CAM. ICCV 2017.