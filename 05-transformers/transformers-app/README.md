### App Structure

```bash
transformers-app/
└── transformer-from-scratch/
    ├── data/
    │   └── tiny_shakespeare.txt
    │
    ├── notebooks/
    │
    ├── models/
    │
    ├── src/
    │   ├── config.py
    │   ├── tokenizer.py
    │   ├── dataset.py
    │   ├── embedding.py
    │   ├── positional_encoding.py
    │   ├── attention.py
    │   ├── multi_head_attention.py
    │   ├── feed_forward.py
    │   ├── transformer_block.py
    │   ├── transformer.py
    │   ├── train.py
    │   ├── generate.py
    │   └── utils.py
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore
```

### Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### Install Packages

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyter notebook tqdm
```

> In my case Apple Silicon Mac (M-series) or recent macOS
