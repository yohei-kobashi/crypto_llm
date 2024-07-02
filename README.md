# crypto_llm

# README.md

## Encrypted Pre-Training for Large Language Models (LLMs)

### Overview

This repository contains the code used for the experiments presented in the short paper on pre-training Large Language Models (LLMs) with encrypted data to address concerns about sensitive data leakage. Our approach involves using a polyalphabetic substitution cipher to encrypt the training data, allowing the LLM to learn abstract language patterns while concealing specific sensitive information.

### Key Components

- **Encryption Method**: We employ a polyalphabetic substitution cipher to encrypt parts of the training data containing sensitive information.
- **Model**: The experiments use the GPT-2 (1.3B) model.
- **Training Framework**: Training is conducted using Megatron-DeepSpeed on NVIDIA H100 GPUs.
- **Data**: The English Wikipedia dump from May 20, 2024, is used, divided into encrypted and plain text segments.

### Files and Directories

- `encryption.py`: Script for encrypting data using a polyalphabetic substitution cipher.
- `train.py`: Main script for training the GPT-2 model with encrypted data.
- `tokenizer.py`: Script for training the tokenizer on both encrypted and plain text data.
- `evaluate.py`: Evaluation scripts to measure model performance in terms of loss and perplexity.
- `data/`: Directory containing the encrypted and plain text data.
- `configs/`: Configuration files for the training and evaluation processes.
- `figures/`: Visual representations of the training methodology and encryption technique used.

### How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Encrypted-LLM-Pretraining.git
   cd Encrypted-LLM-Pretraining
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ and the required packages installed. You can use the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Encrypt the Data**:
   Use the `encryption.py` script to encrypt your data. Modify the script to specify the text and the key length.
   ```bash
   python encryption.py --input data/wikipedia.txt --output data/encrypted.txt --key_length 1000
   ```

4. **Train the Model**:
   Use the `train.py` script to start the training process. Adjust the configuration as needed in the `configs` directory.
   ```bash
   python train.py --config configs/gpt2_config.json
   ```

5. **Evaluate the Model**:
   After training, evaluate the model using the `evaluate.py` script.
   ```bash
   python evaluate.py --model_path checkpoints/ --data_path data/wikitext-2-raw-v1/
   ```

### Results

- Models pre-trained on encrypted data demonstrated effective transfer to natural language tasks after subsequent training on plain texts.
- The impact of encryption strength on model performance was analyzed, revealing that higher cipher strengths can negatively affect training efficiency and model utility.

### Future Work

- Exploring larger models and different encryption techniques to mitigate the impact of high encryption strength.
- Investigating the potential for combining polyalphabetic ciphers with other classical methods like transposition ciphers for improved training outcomes.

### References

- Carlini, N., et al. (2021). Extracting Training Data from Large Language Models. arXiv preprint arXiv:2012.07805.
- Dong, X., et al. (2024). Building Guardrails for Large Language Models. arXiv preprint arXiv:2401.07895.
- Hattenbach, B., & O’Rourke, J. (2015). Patents and the Emergence of a Digital Ocean. Nature Biotechnology.
- Hristov, H. (2016). Artificial Intelligence and Copyright Law: Who Owns the Content Created by AI? The John Marshall Journal of Information Technology & Privacy Law.
- Vyas, Y., et al. (2023). Provable Data Privacy in Machine Learning: A Survey. IEEE Transactions on Neural Networks and Learning Systems.

For more details, please refer to our [paper](#).

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Acknowledgments

The acknowledgements section is intentionally omitted in this version of the README to maintain anonymity during the peer review process. Full acknowledgements will be included in the final publication.