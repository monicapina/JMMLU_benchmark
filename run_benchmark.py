
import os
import re
import time
import pandas as pd
import torch
import transformers
from typing import List, Dict, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from context_generator import generate_context
from umls_rerank_cohere import get_umls_keys
from cohere.errors import TooManyRequestsError
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from torch import cuda
import pdb
from langchain_core.runnables.history import RunnableWithMessageHistory  # Updated import for conversation
import torch

class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        return self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        d = super().load_memory_variables(inputs)
        d.pop("history", None)
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d

class UMLSQuestionAnswering:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", hf_token: str = None):
        # Load the model and tokenizer from Hugging Face

        self.pipe = pipeline("text-generation", model= model_name, max_new_tokens=500, return_full_text=True, device=cuda.current_device() if cuda.is_available() else "cpu")

        self.pipe = HuggingFacePipeline(pipeline=self.pipe)

        memory = ExtendedConversationBufferWindowMemory(k=0,
                                                        ai_prefix="Physician",
                                                        human_prefix="Patient",
                                                        extra_variables=["context"])

        template = """
        <s>[INST] <<SYS>>
        Answer the question in conjunction with the following content.
        <</SYS>>

        Context:
        {context}

        Question: {input}
        Answer: [/INST]
        """

        PROMPT = PromptTemplate(
            input_variables=["context", "input"], template=template
        )
        
        self.conversation = ConversationChain(
            llm=self.pipe,
            memory=memory,
            prompt=PROMPT,
            verbose=True,
        )        

        # Initialize variables for accuracy tracking
        self.csv_files = [
            "professional_medicine.csv",
            "world_history.csv",
            "college_computer_science.csv",
            "management.csv"
        ]
        self.accuracy_per_file = {}
        self.total_questions = 0
        self.correct_answers = 0
    
    def generate_context_with_retries(self, question, PROMPT, max_retries=3, base_wait_time=10, max_wait_time=60):
        """Generates context with retries, using caching and fallback strategies."""
        retries = 0
        while retries < max_retries:
            try:
                context = get_umls_keys(question, PROMPT, self.pipe)
                return context
            except TooManyRequestsError:
                retries += 1
                wait_time = min(base_wait_time * retries, max_wait_time)
                print(f"Rate limit reached. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying...")
                retries += 1
                time.sleep(5)

        # Fallback if all retries fail
        print("Failed after multiple retries. Using fallback context.")
        return "Fallback context due to API limit."
    
    def extract_option(self, generated_answer):
        # Regex pattern to find the first occurrence of an option (A, B, C, or D) followed by a dot or a space
        match = re.search(r'\b([A-D])\b', generated_answer)
        #match = re.search(r"the correct answer is \*\*\((\w)\)", generated_answer, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return None

    
    def ask_question(self, question: str, options: List[str]) -> str:
        # Combine the question and options into a single prompt
        
        PROMPT = """
        [INST] Input: {question}

        Our knowledge graph contains definitions and relational information for medical terminologies. Please:

        1. Analyze the question to determine helpful medical terminologies.
        2. Return 3-5 key medical terminologies in this format:
        {"medical terminologies": ["term1", "term2", ...]}
        3. Include relationships between the terms:
        {"relationships": [{"from": "term1", "to": "term2", "type": "is a"}, ...]}

        Options:
        A. {options[0]}
        B. {options[1]}
        C. {options[2]}
        D. {options[3]}
        Choose the correct option: A, B, C, or D.
        [/INST]
        """      
        
        formatted_question = f"{question}\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nWhat is the correct option? Choose A, B, C, or D."
        
        #context = self.generate_context_with_retries(question, PROMPT) # For API rate limiting
        context = get_umls_keys(question, PROMPT, self.pipe) #KG-Rank-UMLS
        #context = generate_context(prompt, token_cohere) #KG-Rank-WikiPedia
        print("context: ", context)
        answer = self.conversation.predict(context=context, input=formatted_question)
        print("answer: ", answer)        
        selected_option = self.extract_option(answer)

        return selected_option

        
    def _update_accuracy(self, dataset_name, selected_option, correct_answer):
        self.total_questions += 1
        #print(self.accuracy_per_file)
        if dataset_name not in self.accuracy_per_file:
            self.accuracy_per_file[dataset_name] = {"correct": 0, "total": 0}
        self.accuracy_per_file[dataset_name]["total"] += 1
        
        if selected_option == correct_answer:
            self.correct_answers += 1
            self.accuracy_per_file[dataset_name]["correct"] += 1
        
        
    def display_accuracy(self):
        print("\nAccuracy per dataset:")
        for dataset, stats in self.accuracy_per_file.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"{dataset}: {accuracy:.2f}%")
        if self.total_questions > 0:
            total_accuracy = (self.correct_answers / self.total_questions) * 100
            print(f"\nOverall Accuracy: {total_accuracy:.2f}%")
        else:
            print("No questions were answered.")
            
    
    def evaluate_with_umls_context(self, folder_path: str):
        """Evaluates specified CSV files in the provided folder and calculates accuracy."""
        total_accuracy = 0
        datasets_evaluated = 0
        
        for file_name in self.csv_files:
            
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                print(f"File {file_name} not found in the provided folder path: {folder_path}")
                continue
            
            dataset_name = file_name.replace(".csv", "")
            
            # Read the CSV file without headers
            data = pd.read_csv(file_path, header=None)
            correct_answers = 0
            total_questions = 0
            
            for _, row in data.iterrows():
                question = row.iloc[0]
                options = [row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]]
                correct_answer = row.iloc[5]
                if question:
                    # Ask the question, including options
                    llm_response = self.ask_question(question, options)
                    self._update_accuracy(dataset_name, llm_response, correct_answer)
            
            # Calculate accuracy for the dataset
            self.display_accuracy()
 
    
print("Initializing UMLSQuestionAnswering system...")
#model_name = "meta-llama/Llama-3.2-3B"
model_name= "google/gemma-2-2b"
#model_name="google/gemma-2-2b-jpn-it"
qa_system = UMLSQuestionAnswering(model_name=model_name)

folder_path = "data/"
print(f"Evaluating datasets in folder: {folder_path}")
qa_system.evaluate_with_umls_context(folder_path)
