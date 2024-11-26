# JMMLU Benchmark

The JMMLU Benchmark repository provides a comprehensive framework for evaluating Japanese language models using the **Japanese Massive Multitask Language Understanding (JMMLU)** benchmark. This benchmark assesses the performance of large language models across various tasks and domains in Japanese.

---

## Features

- **Multitask Evaluation**: 
  Evaluate language models on a wide range of subjects, including:
  - Professional Medicine
  - World History
  - Computer Science
  - Management
- **UMLS Integration**:
  Incorporate Unified Medical Language System (UMLS) keys to enhance context generation for medical-related queries.
- **Accuracy Tracking**:
  Automatically calculates and displays accuracy metrics for each dataset evaluated.

---

## Installation

### Prerequisites
- **Python**: Version 3.9 or higher
- **`pip`**: For package management
- **GPU (Optional)**: A CUDA-compatible GPU is recommended for optimal performance.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/monicapina/JMMLU_benchmark.git
   cd JMMLU_benchmark
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # For Linux/Mac
   venv\Scripts\activate           # For Windows
   ```

3. **Install Dependencies**:
   Install all required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   Ensure all dependencies are installed successfully:
   ```bash
   python -m pip check
   ```

---

## Usage

### Dataset Preparation
Place your dataset files in the `data/` directory. Each dataset should be a CSV file formatted as follows:

| Question                | Option A       | Option B       | Option C       | Option D       | Correct Answer |
|-------------------------|----------------|----------------|----------------|----------------|----------------|
| Sample question text... | Option 1 text  | Option 2 text  | Option 3 text  | Option 4 text  | A              |

### Run the Evaluation Script
To evaluate the datasets, execute:
```bash
python main.py
```

The script will process each dataset in the `data/` directory and output accuracy metrics for each file.

---

## Project Structure

```plaintext
JMMLU_benchmark/
│
├── data/                   # Directory for dataset CSV files
├── main.py                 # Main script for running evaluations
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation
```

---

## Contributions

Contributions are welcome! If you have suggestions, improvements, or new features, feel free to create a pull request or open an issue on the [GitHub repository](https://github.com/monicapina/JMMLU_benchmark).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support

For questions or issues, please open an issue on the [GitHub Issues page](https://github.com/monicapina/JMMLU_benchmark/issues).

---

## Acknowledgments

Special thanks to contributors and the community for supporting the development and improvement of this benchmark.

---

With this repository, you can effectively evaluate Japanese language models across multiple tasks and improve their performance.
