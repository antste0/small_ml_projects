# small_ml_projects

A collection of small machine learning projects that are too small for their own repository. Each folder has its own README with details and results.

## Projects

| Project | Description | Stack |
|---|---|---|
| [CIFAR-10](image_recognition/README.md) | Image classification (10 classes) with a CNN; the best model reaches 0.81 test accuracy | Keras, NumPy, Matplotlib |
| [Camera image recognition](image_recognition/camera_image_recognition/cnn/README.md) | Real-time object recognition from a camera (bottle, pen, flashlight); 0.911 test accuracy | PyTorch, OpenCV, Hugging Face `datasets` |
| [Markov chains: rock-paper-scissors](markov_chains/README.md) | A Markov chain player against a Bayesian learning player (Dirichlet prior) | Python, Matplotlib |
| [RAG](rag/README.md) | Retrieval-Augmented Generation: vector index search and answer generation | FAISS, `sentence-transformers`, Transformers |

## About the projects

### [CIFAR-10](image_recognition/README.md)

Two small CNNs (131k and 141k parameters) trained on CIFAR-10 with data augmentation, `EarlyStopping` and `ReduceLROnPlateau`. The comparison shows that a shallower, wider network with GELU (3 Conv blocks, 0.81 accuracy) beats a deeper one with ReLU (5 blocks, 0.78) despite a similar parameter count. The most common errors are cat/dog and bird/frog confusions.

### [Camera image recognition](image_recognition/camera_image_recognition/cnn/README.md)

A complete pipeline from data to a live demo. `model.ipynb` loads the dataset, applies letterbox preprocessing (preserving the aspect ratio) and trains a CNN with `SiLU`, `BatchNorm` and `Dropout`. `main.ipynb` loads the best checkpoint (0.84 MB) and classifies frames from a webcam. It reaches 0.911 accuracy on the test set.

### [Markov chains: rock-paper-scissors](markov_chains/README.md)

A simulation of a game between two players. One plays according to a fixed transition matrix (the next move depends on the previous one), the other learns that matrix online from observed moves using counts with a Dirichlet prior. After 1000 rounds the learner wins 477 rounds against 248 losses, and its estimated matrix is close to the true one.

### [RAG](rag/README.md)

A question-answering system over a single document. Text fragments are embedded with `multilingual-e5-base`, indexed in FAISS, and the retrieved context is passed to a quantised `Qwen2.5-3B-Instruct` (a CUDA GPU is required). The model refuses to answer questions outside the document, and when it finds nothing it retries retrieval with a larger `k`. The evaluation is qualitative only.
