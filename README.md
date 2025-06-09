# Neural Machine Translation Project

This project implements three different Neural Machine Translation (NMT) systems:

1. **Transformer Model** — English to Spanish
2. **LSTM Encoder-Decoder Model** — English to Italian
3. **Sequence-to-Sequence (Seq2Seq) with Bahdanau Attention** — English to Spanish

---

## Reproducing the Results

### 1. Environment Setup

The project was developed using:

- **Python**: 3.11.13
- **PyTorch**: 2.6.0+cu124
- **NumPy**: 2.0.2
- **Matplotlib**: 3.10.0
- **Transformers (HuggingFace)**: 4.x (for the Transformer model)

No need to worry about this since each document contains all the important downloads and imports you must do to run correctly.
---

### 2. Datasets


- **Transformer**:  
  - **Dataset**: [sentence-transformers/parallel-sentences-opensubtitles](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles)  
  - Loaded directly from Hugging Face datasets.
  
- **LSTM Encoder-Decoder**:  
  - **Dataset**: [English-Italian Parallel Corpus (`eng_ita_v2.txt`)](https://raw.githubusercontent.com/kyuz0/llm-chronicles/main/datasets/eng_ita_v2.txt)  
  - Automatically downloaded from GitHub using `wget` when running the notebook.

- **Seq2Seq + Attention**:  
  - **Dataset**: [English-Spanish Translation Dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset)  
  - **Filename**: `data.csv`

---

### 3. Running the Notebooks

The code is designed to run on **Google Colab** with GPU acceleration.

---

### 3.1 Transformer (English to Spanish)

To test this model, open the `Transformer.ipynb` file. The only model worth testing is the final transformer saved under the Hugging Face account as `my_model_f`.  We have provided some English sentences for testing purposes.

#### Steps:

1. Open the notebook `Transformer.ipynb` in Colab.
2. Load the pre-trained model available under the name `my_model_f` and test translations on the provided sentences by running the last cell of the notebook to load the model .

**Note:**  
While the model translates basic sentences from English to Spanish, performance is limited. For some sentences different from the ones given, translations may be incorrect or fail completely.

---

### 3.2 LSTM Encoder-Decoder (English to Italian)

#### Steps:

1. Open the notebook `LSTM_encoder_decoder.ipynb` in Colab.
2. Run all cells up to the training part.
3. To load and test the final model, go to the **Load and Test** section and run those cells to load the pretrained encoder and decoder weights.
4. In the final cell, test translations by inputting your own sentence (must use words present in the vocabulary).

Example sentences you can try:
- `she likes music`
- `listening to music`
- `tom left yesterday`
- `i want to go home now`
- `he likes chocolate`
- `tom was right about that`
- `tom said he would not come`

**Note:**  
Only words present in the training vocabulary are supported (e.g., the name `tom` is the only name included). Inputting unknown words will cause the model to fail.

---

### 3.3 Seq2Seq with Attention (English to Spanish)
This notebook implements a Sequence-to-Sequence (Seq2Seq) architecture with Bahdanau Attention, trained on an English-Spanish parallel corpus.

#### Steps:

1. Open the notebook `TranslationWSeq2Seq&Attention.ipynb` in Colab.
2. Upload the `data.csv` file to your Colab environment.
3. Set the `results_path` and `data_path` accordingly.
4. Trim the data as desired using the `prepareData` function. The dataset is large, and not trimming it can lead to very long training times.
5. Run the notebook cells sequentially to train the model or load pretrained embeddings.

**Note:**  
If using Word2Vec pretrained embeddings, it is recommended to:

```bash
!pip uninstall -y gensim numpy scipy
!pip install numpy==1.23.5 scipy==1.10.1 gensim==4.3.1

```
Reset the Colab runtime to apply the changes.
