# CRNN — License-Plate Recognition

End-to-end pipeline built around a **T**hin-**P**late-Spline (TPS) Spatial Transformer, a ResNet backbone, Bi-LSTM sequence modeling, and interchangeable **Attention** or **CTC** prediction heads.

---

## 1  Clone the repository

```bash
git clone https://github.com/HermanLKH/Automatic-Malaysia-Car-License-Plate-Detection-and-Recognition.git
cd Automatic-Malaysia-Car-License-Plate-Detection-and-Recognition/CRNN
```

## 2  Set up Python 3.13 + virtual-env

| step                     | Windows                                                                                       | macOS / Linux                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Download Python 3.13** | [ python-3.13.0-amd64.exe ](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe) | [ 3.13.0 release page ](https://www.python.org/downloads/release/python-3130/) |
| **Create venv**          | `py -3.13 -m venv virt`                                                                       | `python3.13 -m venv virt`                                                      |
| **Activate venv**        | `virt\Scripts\activate`                                                                       | `source virt/bin/activate`                                                     |
| **Install deps**         | `pip install -r requirements.txt`                                                             | same                                                                           |

Note: CUDA wheels for PyTorch are pulled automatically if a supported GPU is detected.

## 3 Grab checkpoints & datasets

Everything is hosted on HuggingFace:
```bash
https://huggingface.co/chinzunyang/COS30018_Best_CRNN
```

Download and drop the three folders at the same level as the notebooks:
```bash
CRNN/
├── pretrained_model/       # official SynthText weights
│   └── TPS-ResNet-BiLSTM-*.pth
├── trained_model/          # your fine-tuned checkpoints
│   └── best_ctc_crnn_!_6939.pth
├── v4_lmdb_data_!/         # train / val / test LMDBs
│   ├── train/
│   ├── val/
│   └── test/
└── notebooks/
    ├── train_crnn_attention_with_!_augment.ipynb
    └── test_crnn_attention_with_!_augment.ipynb

```

## 4 Run the notebooks

| notebook                                        | what it does                                                                                                           |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **`train_crnn_attention_with_!_augment.ipynb`** | trains the full TPS-ResNet-BiLSTM-**Attn** model with data augmentation; best checkpoint is saved to `trained_model/`. |
| **`test_crnn_attention_with_!_augment.ipynb`**  | loads the saved weight file and reports **sample-level** and **character-level** accuracy on the held-out test LMDB.   |

## 5 Questions?

Open an issue in the repo or ping 102781103@students.swinburne.edu.my. Happy hacking! 🚀



