# Databricks notebook source
# MAGIC %md
# MAGIC # NLTK Downloads Logging with MLflow
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC This notebook, presents a tested and effective method on `Runtime DBR 14.1 ML` for enhancing ML workflows using NLTK and MLflow. It specifically addresses the streamlined logging of NLTK resources, like corpora and taggers, as MLflow artifacts. 
# MAGIC
# MAGIC ### Author Details
# MAGIC - **Name**: Debu Sinha
# MAGIC - **Email**: debu.sinha@databricks.com
# MAGIC
# MAGIC In this example, we demonstrate the efficient logging of NLTK resources as MLflow artifacts and how these pre-logged resources can be utilized during inference. This approach significantly expedites the deployment process for real-time inference endpoints, offering both efficiency and reliability.
# MAGIC
# MAGIC ## Advantages of Pre-Logging NLTK Resources
# MAGIC
# MAGIC The traditional approach of downloading NLTK resources at runtime, especially during the deployment phase, presents several challenges:
# MAGIC
# MAGIC 1. **Time Efficiency**: Downloading resources during each deployment can be notably time-consuming, more so for larger corpora or multiple resources.
# MAGIC
# MAGIC 2. **Dependency and Reliability**: This method creates a reliance on external services (e.g., NLTK servers), which introduces potential failure points.
# MAGIC
# MAGIC 3. **Resource Utilization**: Repeatedly downloading the same resources for multiple deployments results in inefficient use of network and computational resources.
# MAGIC
# MAGIC ### The Benefits of Logging with MLflow
# MAGIC
# MAGIC Contrastingly, pre-logging NLTK resources as MLflow artifacts offers significant improvements:
# MAGIC
# MAGIC - **Speed and Efficiency**: Bundling all necessary resources with the model artifact removes the need for runtime downloads, streamlining deployment.
# MAGIC
# MAGIC - **Reliability**: This method decreases dependency on external services during deployment, enhancing the overall reliability of the process.
# MAGIC
# MAGIC - **Consistency in Model Behavior**: It ensures that the same version of resources is used consistently across training and inference phases.
# MAGIC
# MAGIC In the following sections, we will guide you through the implementation of this approach, illustrating how it can optimize and refine the deployment of ML models in real-world scenarios.
# MAGIC

# COMMAND ----------

# Import required libraries
import mlflow
import nltk
import mlflow.pyfunc
import pandas as pd

# Define a custom NLTK model class extending mlflow.pyfunc.PythonModel
class NLTKModel(mlflow.pyfunc.PythonModel):
    # Method to load and set up the NLTK context
    def load_context(self, context):
        # Retrieve the path to NLTK data from logged artifacts
        nltk_data_path = context.artifacts["nltk_data"]
        # Append the nltk_data_path to nltk's data path
        nltk.data.path.append(nltk_data_path)

    # Method for model prediction
    def predict(self, context, model_input):
        # Tokenize each sentence in the model_input
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in model_input['sentences']]

        # Tag each tokenized sentence with POS tags
        tagged_sentences = [nltk.pos_tag(tokens) for tokens in tokenized_sentences]

        # Format the tagged sentences into a string representation
        formatted_sentences = [' '.join([f"{word}_{tag}" for word, tag in sentence]) for sentence in tagged_sentences]

        # Return the formatted sentences
        return formatted_sentences

# Function to download necessary NLTK datasets
def download_nltk_data():
    # Download 'wordnet', 'averaged_perceptron_tagger', and 'punkt' datasets from NLTK
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')

# Function to log NLTK artifacts and the custom model using MLflow
def log_nltk_artifacts():
    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Download the necessary NLTK data
        download_nltk_data()
        # Create an instance of NLTKModel
        model = NLTKModel()
        # Prepare a sample input for the model
        sample_input = pd.DataFrame({"sentences": ["This is a sample sentence.", "Another example sentence."]})
        # Infer the input and output signature of the model
        signature = mlflow.models.infer_signature(sample_input, model.predict(None, sample_input))
        # Log the model with MLflow, including its signature and a sample input
        mlflow.pyfunc.log_model("custom_nltk_model", 
                                python_model=model,
                                artifacts={'nltk_data': '/root/nltk_data'},
                                signature=signature,
                                input_example=sample_input)

        # Return the run ID for future reference
        return run_id

# Execute the artifact logging function and print the run ID
run_id = log_nltk_artifacts()
print(f"Run ID: {run_id}")


# COMMAND ----------

import mlflow

logged_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/custom_nltk_model')
sample_input = pd.DataFrame({"sentences": ["This is a sample sentence.", "Another example sentence."]})
logged_model.predict(sample_input)
