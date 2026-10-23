# RAG

Retrieval-Augmented Generation: vector index search and answer generation. Code: [`rag.ipynb`](rag.ipynb).

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
