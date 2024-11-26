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


### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/monicapina/JMMLU_benchmark.git
   cd JMMLU_benchmark
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   conda create -n jp_benchmark python=3.9 -y
   conda activate jp_benchmark
   ```

3. **Install Dependencies**:
   Install all required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```


---

## Usage

### Dataset Preparation
The dataset files are located in the data/ directory. The JMMLU dataset is a four-choice question set designed to evaluate the performance of large language models in Japanese. Each dataset is stored in CSV format, and each row represents a question with the following structure:

| Question                | Option A       | Option B       | Option C       | Option D       | Correct Answer |
|-------------------------|----------------|----------------|----------------|----------------|----------------|
| Sample question text... | Option 1 text  | Option 2 text  | Option 3 text  | Option 4 text  | A              |

### Run the Evaluation Script
To evaluate the datasets, execute:
```bash
python run_benchmark.py
```

The script will process each dataset in the `data/` directory and output accuracy metrics for each file.

---

## Project Structure

```plaintext
JMMLU_benchmark/
│
├── data/                   # Directory for dataset CSV files
├── run_benchmark.py        # Main script for running evaluations
├── umls_rerank_cohere.py   # KG-Rank-UMLS
├── context_generator.py    # KG-Rank-WikiPedia                    
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation
```

---

## Error Handling

During development and execution, certain errors may occur. Refer to the [Error Logs](https://github.com/monicapina/JMMLU_benchmark/blob/developed/error_logs.md) file for complete details and solutions.