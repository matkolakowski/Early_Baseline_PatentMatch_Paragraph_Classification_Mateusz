import json
from typing import Tuple, List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
import mlflow
from . import utils
import numpy as np

class TextSimilarityDataset(Dataset):
    """
    A PyTorch Dataset class for loading text similarity data.

    Attributes:
        data (List[Tuple[str, str, int]]): A list of tuples containing pairs of texts and their similarity label.
        tokenizer: A tokenizer object to convert text into a format suitable for the model.
        max_length (int): The maximum length of the tokenized text sequences.
    """

    def __init__(self, data: List[Tuple[str, str, int]], tokenizer: AutoTokenizer, max_length: int):
        """
        Initializes the TextSimilarityDataset with data, tokenizer, and maximum sequence length.

        Args:
            data (List[Tuple[str, str, int]]): A list of tuples containing pairs of texts and their similarity label.
            tokenizer (AutoTokenizer): A tokenizer object to convert text into a format suitable for the model.
            max_length (int): The maximum length of the tokenized text sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves an item from the dataset by index.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing tokenized input IDs,
            attention masks, token type IDs, and the similarity label as tensors.
        """
        text_a, text_b, label = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()
        return input_ids, attention_mask, token_type_ids, torch.tensor(label, dtype=torch.long)


class TextSimilarityLLMManager:
    """
    A manager class for training and evaluating a text similarity model using transformers.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        tokenizer_name (str): The name of the tokenizer to use.
        config (Dict[str, Any]): A configuration dictionary with training parameters.
        MLFlow_reporting (bool): Whether to report metrics to MLFlow.
        verbose (bool): Whether to print out verbose messages during training and evaluation.
    """

    def __init__(self, model_name: str, tokenizer_name: str, config: Dict[str, Any], MLFlow_reporting: bool,
                 verbose: bool):
        """
        Initializes the TextSimilarityLLMManager with a model name, tokenizer name, configuration, and reporting options.

        Args:
            model_name (str): The name of the pre-trained model to use.
            tokenizer_name (str): The name of the tokenizer to use.
            config (Dict[str, Any]): A configuration dictionary with training parameters.
            MLFlow_reporting (bool): Whether to report metrics to MLFlow.
            verbose (bool): Whether to print out verbose messages during training and evaluation.
        """
        self.model_name = model_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'], no_deprecation_warning=True)
        self.MLFlow_reporting = MLFlow_reporting
        self.verbose = verbose

        if verbose:
            if str(self.device) == 'cpu':
                print(f'GPU not found, set GPU for training process')
            elif str(self.device) == 'cuda':
                print(f'GPU found!')


    def load_train_data(self, train_path):
        """
        Loads and preprocesses data, creating DataLoaders for training and validation.

        Returns:
            DataLoader: A DataLoader object.
        """
        # open json
        with open(train_path, 'r') as file:
            data = json.load(file)

        # split data to train and validation splits
        train_data, val_data = train_test_split(
            data, test_size=self.config['test_size'], random_state=self.config['random_state']
        )

        # tokenize combined strings with special ['SEP'] tokens
        train_dataset = TextSimilarityDataset(train_data, self.tokenizer, self.config['max_length'])
        val_dataset = TextSimilarityDataset(val_data, self.tokenizer, self.config['max_length'])

        # create a torch dataloader object
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        self.validation_dataloader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        if self.verbose:
            print(f'Train data len: {len(train_data)}')
            print(f'Validation data len: {len(val_data)}')



    def load_test_data(self, test_path) -> DataLoader:
        """
        Loads and preprocesses test data, creating DataLoader for evaluation.

        Returns:
            DataLoader: A DataLoader object.
        """

        # open json
        with open(test_path, 'r') as file:
            data = json.load(file)

        # tokenize combined strings with special ['SEP'] tokens
        test_dataset = TextSimilarityDataset(data, self.tokenizer, self.config['max_length'])

        # create a torch dataloader object
        test_dataloader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True)

        return test_dataloader

    def train(self):
        """
        Trains the model using the provided training DataLoader and evaluates it using the validation DataLoader.

        Args:
            train_dataloader (DataLoader): The DataLoader for training data.
            validation_dataloader (DataLoader): The DataLoader for validation data.
        """

        if self.MLFlow_reporting:
            for param_name, param_value in self.config.items():
                mlflow.log_param(param_name, param_value)

        for epoch in range(self.config['num_epochs']):
            self.model.train()
            train_loss = 0.0
            for batch in self.train_dataloader:
                batch = [item.to(self.device) for item in batch]
                b_input_ids, b_attention_mask, b_token_type_ids, b_labels = batch

                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids, labels=b_labels)
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = train_loss / len(self.train_dataloader)

            epoche_metrics, cm = self.evaluate(self.validation_dataloader, phase='Validation', epoch=epoch)

            if self.MLFlow_reporting:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)



    def evaluate(self, dataloader: DataLoader, phase: str = 'Test', epoch: int='None'):
        """
        Evaluates the model using the provided DataLoader.

        Args:
            dataloader (DataLoader): The DataLoader for evaluation data.
            phase (str): The phase of evaluation (e.g., 'Validation', 'Test').
        """
        self.model.eval()
        predictions, true_labels = [], []
        for batch in dataloader:
            batch = [item.to(self.device) for item in batch]
            b_input_ids, b_attention_mask, b_token_type_ids, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids)

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            true_labels.extend(b_labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)

        metrics_dict = {"accuracy":accuracy, "f1":f1, "mcc":mcc}
        if epoch is not None:
            cm_filename = f"artifacts/{utils.timestamp()}confusion_matrix_epoch_{epoch}.csv"
        else:
            epoch = 0
            cm_filename = f"artifacts/{utils.timestamp()}confusion_matrix.csv"
        np.savetxt(cm_filename, cm, delimiter=",")


        if self.verbose:
            print(f"{phase} Accuracy: {accuracy}")
            print(f"{phase} F1 Score: {f1}")
            print(f"{phase} Matthews Correlation Coefficient: {mcc}")
            print(f"{phase} Confusion Matrix:\n{cm}")

        if self.MLFlow_reporting:

            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
            np.savetxt(cm_filename, cm, delimiter=",")
            mlflow.log_artifact(cm_filename)

        return metrics_dict, cm


    def run(self, test_path: str, train_path: Optional[str] = None):
        """
        The main method to run the training and evaluation process.

        Args:
            test_path (str): The path to the test data file.
            train_path (Optional[str]): The path to the training data file. If not provided, only evaluation is performed.
        """
        if train_path is not None:
            self.load_train_data(train_path)
            self.train()

        test_dataloader = self.load_test_data(test_path)
        self.evaluate(test_dataloader, 'Test')