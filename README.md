# Low-Data Breast Cancer Classification (TensorFlow)

Built TensorFlow MLP models to classify breast cancer under low-data settings (5% and 10% training splits) and compared two network architectures with reproducible training.

## Project Files
- `prog1-5.py`  : Model 1 (5% training)
- `prog1-10.py` : Model 1 (10% training)
- `prog2-5.py`  : Model 2 (5% training)
- `prog2-10.py` : Model 2 (10% training)
- `documentation.pdf` : Project write-up / results

## Results (Accuracy)
- 5% training:
  - Prog1-5: 0.9192
  - Prog2-5: 0.9016
- 10% training:
  - Prog1-10: 0.9315
  - Prog2-10: 0.9297

## Requirements
```bash
pip install -r requirements.txt
