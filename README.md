# Business Requests Parsing

## Overview
This repository contains the tool to parse business requirements document (BRD) and agreement aprovals under a directory of business requests.
The tool is essentially a pipeline that processes data inside PDFs and Excel Files, applys embeddings, and stores them in a Qdrant Databse.
The business requirements document pipeline also generates a summary for each business request and stores them in a seperate vector database.

## Installing Dependencies (Developed Under Python 3.11.4)
   ```
   pip install -r requirements.txt
   ```
## Notes
   - **Change the inferencing model to local models once available (See BR_query_engine.py and https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)**
   - **Ensure that files and folders are named properly (see images in Usage section on examples of proper structured directory)**
   - **Older business requests before BR40000s might not work as expected due to directory structure inconsistency**
   - **French data is excluded for simplicity but can easily be added by changing the pattern(Business Requirements Document) to identify a BRD and changing the metadata of the extracted nodes**
   - **Parsed PDFs are not as accurate as parsed Excel files due to tool inability**
   - **Dupilicated data may be stored in the directory due to data inconsistency**
   - agreement_ingestion.py contains the code to parse agreement approvals
   - BRD_ingestion.py contains the code to parse BRDs and generate summaries based of these BRDs
   - utils.py contains common utility functions shared by some of files

## Usage
**Make sure to set your environment variable OPENAI_API_KEY to your OpenAI API key so that the summary of each business request can be generated.**  
**Make sure that Qdrant is running in your local host.**  
**Pass in different parameters in the configuration file as needed.**  
  
```
python3 file_name
```  
  
After parsing, you can visit Qdrant to check your saved data.  
  
![alt text](https://github.com/ssc-dsai/Business_Requests_Parsing/blob/main/example1.png)  
**A directory of business requests should look like the image above (exact same namings)**

![alt text](https://github.com/ssc-dsai/Business_Requests_Parsing/blob/main/example2.png)  
**A directory of a business request should look like the image above (exact same namings)**


## Example Usage
```
python3 BRD_ingestion.py
```

This command parses all the Business Requirements Document

## Configurations
1. **Available Tools**
   - BRD_ingestion.py (To extract BRDs and generate summaries for each business request)
   - agreement_ingestion.py (To extract agreement approvals)

2. Modify config.json as needed to change the behaviour of the program
   
|Parameter                 |Description                                                                               |Options                                                          |
|--------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
|model                     |This is the OpenAI Generative Model that will be used to summarize the BRDs               |See https://platform.openai.com/docs/models for different models |
|source_folder_path        |This is the path to where the the business requests are stored                            |Your choice                                                      |
|embedding_cache_directory |This is the path to where the embedding model is store                                    |This is generated by default                                     |
|summary_collection_name   |This is the name of the index that stores the summary of the BRDs                         |Your choice                                                      |
|context_collection_name   |This is the name of the index that stores the context of the BRDs                         |Your choice                                                      |
|agreement_collection_name |This is the name of the index that stores the agreement approval of the business requests |Your choice                                                      |  

## To-Do
   - **Use AI agent to look at document similarity and deal with inconsistent directory structure(this will solve many problems like duplicated data, outdated data, and corrupted metadata)**
   - Need a better tool/library to extract data from tables inside PDFs
   - Better chunking technique?
