# GP-WAITER

GP-WAITER (Genome-Phenotype prediction using Weighted self-AttentIon TransformER) is a model that integrates GWAS-derived SNP weights into a CNN/Transformer-style pipeline for genotype-to-phenotype prediction.

## Windows compatibility

The old repo shipped a Linux-only compiled extension (`*.so`). This repo now runs from pure Python (`model/GP_WAITER.py`), so it works on **Windows / Linux / macOS** with the same codebase.

The Linux prebuilt `.so` artifacts have been removed from version control; runtime now uses the source model implementation only.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train on your own data

```bash
python train-test.py \
  --genotype-txt /path/to/data.genotype.txt \
  --phenotype-csv /path/to/data.phenotype.csv \
  --weight-csv /path/to/data.weighted.csv \
  --phenotype-column O \
  --rows 112 \
  --cols 597
```

### 3) Run the demo

```bash
cd "Demo/Instructions to run on data"
python demo.script.py
```

## Windows helpers

You can also use these wrappers from the repository root:

- `run_demo_windows.bat`
- `run_train_windows.bat`

## Notes

- GPU is used automatically when available; otherwise CPU is used.
- The fallback model API is compatible with existing code that instantiates `TModel(embed_size, w, param, num_layers)`.
