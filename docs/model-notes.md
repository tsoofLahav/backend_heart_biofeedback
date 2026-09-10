# Model provenance

The overview describes the three models developed across the practicum, while the architecture flow describes the current backend at `2ee1698`. These are different scopes: the latest route does not run all three networks in sequence.

- **Classifier:** historical `classifier.py` defines dense layers input → 64 → 32 → 32 → 1 with ReLU and sigmoid. The archived training script `models/UnetModel/train_MLP.py` trains the window classifier. The quality diagram uses a 48-sample, two-second window at 24 Hz. Its output is a quality decision, not a missing waveform; masking is downstream logic.
- **Reconstruction:** historical `reconstruction.py` and archived `models/UnetModel/train_Unet.py` define 240 → 512 → 256 → 128 → 48 with ReLU and dropout. Training masks two seconds in clean ten-second examples and minimizes MSE against the removed samples. Dataset inputs include a BIDMC-derived signal archive. The diagram depicts the intended replacement of the missing segment, not a claim that all historical integration paths were verified.
- **Prediction:** current `predict_model.py` and archived `models/TCN/train_tcn_model.py` define flatten → 16 → 128 → 64 → 8 with ReLU. The training implementation uses Smooth L1 loss. Despite the filenames and older README, these definitions contain neither temporal convolutions nor a U-Net.

[Historical classifier](https://github.com/tsoofLahav/backend_hearmonitor/blob/1236c1f1eb735237659fba38dae9a37ece6d75c8/classifier.py) · [Historical reconstruction](https://github.com/tsoofLahav/backend_hearmonitor/blob/1236c1f1eb735237659fba38dae9a37ece6d75c8/reconstruction.py) · [Current predictor](../predict_model.py)

Training descriptions were checked against the author's locally archived practicum scripts. Those scripts and datasets are not included in this backend checkout. No training-set sizes or quantitative model-performance claims are inferred from the illustration.

The current runtime interpolates capture gaps, performs conventional filtering and readability checks, and invokes the predictor. It can extend a short forecast using the mean observed interval and adjusts boundaries between audio chunks. Not every emitted cue is therefore a direct neural-network output. The fixed forecast window is intended to accommodate latency; this implementation does not measure and compensate every component of device and network delay dynamically.
