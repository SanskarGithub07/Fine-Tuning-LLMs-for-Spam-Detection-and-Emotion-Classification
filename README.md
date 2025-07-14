# Fine-Tuning-LLMs-for-Spam-Detection-and-Emotion-Classification

This project explores fine-tuning a lightweight causal language model (**SmolLM2-135M**) for two downstream classification tasks:

* **Spam Detection** on a short SMS dataset
* **Emotion Classification** on user-generated text

Both tasks use custom classification heads on top of the pretrained language model.

---

### Final Results

| Task                | Dataset Size | Classes                   | Final Test Accuracy |
| ------------------- | ------------ | ------------------------- | ------------------- |
| Spam Classification | 700+         | Spam, Ham                 | **86.33%**          |
| Emotion Detection   | 14,000+      | Happy, Sad, Angry, Others | **77.23%**          |

---

### Technologies Used

* **Model**: `SmolLM2-135M` (causal language model)
* **Libraries**: PyTorch, Hugging Face Transformers, Pandas, NumPy, scikit-learn
* **Evaluation**: Accuracy, Confusion Matrix

---

### How It Works

1. **Preprocessing**

   * Cleaned and tokenized text inputs
   * Balanced datasets using undersampling
   * Padded/truncated sequences with `Tokenizer.pad()`

2. **Model Fine-Tuning**

   * Added classification head (`Linear`) on top of SmolLM2
   * Used `BCEWithLogitsLoss` (binary) and `CrossEntropyLoss` (multi-class)
   * Fine-tuned with Adam optimizer for multiple epochs

3. **Evaluation**

   * Accuracy computed on the test set
   * Confusion matrices plotted for both models

### Notes

* The model uses `SmolLM2-135M` for faster training and lower resource consumption.
* All experiments are run on google colab and kept modular for easy extension.

---

