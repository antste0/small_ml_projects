# small_ml_projects

A collection of small machine learning projects that are too small for their own repository.

## Projects

| Folder | Description | Stack |
|---|---|---|
| [`image_recognition/cifar-10.ipynb`](image_recognition/cifar-10.ipynb) | CIFAR-10 image classification (10 classes) with a CNN using BatchNorm, MaxPooling, GlobalAveragePooling and Dropout | Keras, NumPy, Matplotlib |
| [`image_recognition/camera_image_recognition/cnn`](image_recognition/camera_image_recognition/cnn) | Real-time object recognition from a camera (bottle, pen, flashlight) | PyTorch, OpenCV, Hugging Face `datasets` |
| [`markov_chains/markov_chain_rps.ipynb`](markov_chains/markov_chain_rps.ipynb) | Rock-paper-scissors: a Markov chain player against a Bayesian learning player (Dirichlet prior) | Python, Matplotlib |
| [`rag/rag.ipynb`](rag/rag.ipynb) | Retrieval-Augmented Generation: vector index search and answer generation | FAISS, `sentence-transformers`, Transformers |

### CIFAR-10

A comparison of two small CNNs trained with data augmentation, `EarlyStopping` (patience 10, best weights restored) and `ReduceLROnPlateau`. Results come from `cifar-10.ipynb`; the test set is also used as the validation set:

| Model | Architecture | Parameters | Epochs | Accuracy (test) | Loss (test) |
|---|---|---|---|---|---|
| 1 | 3 Conv blocks (64-64-128), GELU | 131k | 30 | **0.81** | 0.56 |
| 2 | 5 Conv blocks (32-64-64-64-64), ReLU | 141k | 26 (early stopping) | 0.78 | 0.65 |

The shallower and wider network with GELU performed better than the deeper one, despite having fewer parameters.

![Model 1 training curves](image_recognition/assets/cifar_training_curves.png)
![val_accuracy comparison of models 1 and 2](image_recognition/assets/cifar_model_comparison.png)

Confusion matrix of model 1 on the test set (10,000 images, 8,132 correct). Cats and dogs are confused most often (106 cats predicted as dogs and 125 dogs as cats), as are birds and frogs (122 birds predicted as frogs). Frogs (938) and automobiles (932) are recognised best.

![Model 1 confusion matrix](image_recognition/assets/cifar_confusion_matrix.png)

### Camera image recognition

- `model.ipynb`: loading the dataset (`imagefolder`), letterbox preprocessing (preserving the aspect ratio), CNN training (`SiLU`, `BatchNorm`, `Dropout`), confusion matrix.
- `main.ipynb`: loads the trained weights from `assets/best_model.pth` and classifies the camera image (`cv2.VideoCapture(0)`).
- Result of the best checkpoint (chosen by the lowest val loss) on the test set: **accuracy 0.911**, loss 0.225. The model's parameter size is 0.84 MB. Training ran for 100 epochs (Adam, lr 0.0012, `ReduceLROnPlateau` lowered the lr to 0.0006 and then to 0.0003).
- Dependencies are listed in `pyproject.toml` (Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/)).

![Training curves](image_recognition/camera_image_recognition/cnn/assets/training_curves.png)
![Confusion matrix](image_recognition/camera_image_recognition/cnn/assets/confusion_matrix.png)

### Markov chains: rock-paper-scissors

The `MarkovPlayer` class plays according to a given transition matrix. `BayesianLearningPlayer` learns this matrix from the opponent's observed moves (counts with a Dirichlet prior) and predicts the next move.

Result of a single simulation (1000 rounds, `alpha=1.0`): the learning player won 477 rounds, lost 248, and 275 ended in a draw. The final score from the Markov player's perspective is −230. For the first 100 rounds the score hovers around 0, between rounds 175 and 300 it stays at around −20, and after that the learning player's lead grows almost linearly. The learned transition matrix is close to the given one (e.g. after `S` the opponent plays `R` with probability 0.79, against 0.8 in the given matrix).

Notation: R is rock, P is paper, S is scissors. The row is the opponent's previous move and the column is the next one.

| After move | R | P | S |
|---|---|---|---|
| R | 0.31 | 0.38 | 0.32 |
| P | 0.41 | 0.16 | 0.43 |
| S | 0.79 | 0.12 | 0.09 |

![Cumulative score](markov_chains/assets/cumulative_score.png)

The result depends on randomness (no fixed seed), so later runs will give different numbers.

### RAG

- Embeddings: `intfloat/multilingual-e5-base`
- Generative model: `Qwen/Qwen2.5-3B-Instruct` (quantised with `BitsAndBytesConfig`, so a CUDA GPU is required)
- Vector index: FAISS

Example answers (corpus: an article about the Union of Poor Peasants from Huliaipole):

| Question | Answer |
|---|---|
| Who was the main supplier of the group's weaponry? | Oleksandr Semenyuta was the main supplier of the group's weaponry. |
| What happened to the union of poor peasants? | The Union of Poor Peasants was deposed in a coup by Ukrainian nationalists, and the town fell under the control of the Austro-Hungarian Army. |
| At what temperature does the water boil? | No information in the document. |

The model correctly refuses to answer questions outside the document (water, the meaning of life). When the answer is "No information", the system repeats the retrieval with a larger `k` (4 → 7). The evaluation is qualitative only: there is no test set with reference answers.
