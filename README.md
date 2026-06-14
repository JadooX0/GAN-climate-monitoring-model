# GAN-climate-monitoring-model
Core Architecture & Pipeline:


Data Pipeline: Uses an automated xarray and dask multi-file stream engine to ingest, align, scale, and clean independent spatial time-series climate files into a structured continuous tensor data cube.

The Generator: Built on a custom block-style U-Net topology that extracts complex spatial atmospheric features and maps them directly to high-fidelity localized irradiance patterns.

The Discriminator: A spatial PatchGAN architecture that evaluates localized structural variations against authentic Copernicus ground-truth baselines to penalize spatial artifact noise.

Statistical Verification: Integrates automated post-training dimensionality reduction via PCA (Latent Variance Alignment) and t-SNE (High-Dimensional Density Clustering) to visually confirm mathematical alignment between synthetic outputs and historical atmospheric realities.

dependencies:


pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118; pip install xarray netcdf4 dask toolz scikit-learn matplotlib numpy

run command:


python drogan_final_pipeline.py 
