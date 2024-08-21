# Business Requests Parsing

## Overview
This repository contains the tool to parse business requirements document and agreement aprovals under a directory of business requests.
The tool is essentially a pipeline that processes data inside PDFs and Excel Files, applys embeddings, and stores them in a Qdrant Databse.
The business requirements document pipeline also generates a summary for each business request and stores them in a seperate vector database.

## Installing Dependencies (Developed Under Python 3.11.4)
   ```
   pip install -r requirements.txt
   ```
## Notes
   - **Ensure that files and folders are named properly (see images in Usage section on examples of proper structured directory)**
   - **Older business requests before BR40000s might not work as expected due to directory structure inconsistency**
   - **French data is excluded for simplicity but can easily be added by changing a few keywords in the code**
   - **Parsed PDFs are not as accurate as parsed Excel files due to tool inability**
   - agreement_ingestion.py contains the code to parse agreement approvals
   - BRD_ingestion.py contains the code to parse BRDs and generate summaries based of these BRDs
   - utils.py contains common utility functions shared by some of files

## Usage
Make sure to set your environment variable OPENAI_API_KEY to your OpenAI API key so that the summary of each business request can be generated.  
  
```
python3 file_name_of_tool path_of_BR_directory BR_summary_collection_name BR_details_collection_name
```  
Passing in different command line arguments allows different functionalities  
  
![alt text](https://github.com/ssc-dsai/Business_Requests_Parsing/blob/main/example1.png)  
**A directory of business requests should look like the image above (exact same namings)**

![alt text](https://github.com/ssc-dsai/Business_Requests_Parsing/blob/main/example2.png)  
**A directory of a business request should look like the image above (exact same namings)**


## Example Usage
```
python3 BRD_ingestion.py ./Output BR_summaries BR_Data
```

This command parses all the Business Requirements Document under the Output directory and stores it in a Qdrant collection named "BR_Summaries" and "BR_Data"

## Command Line Options
1. **Available Tools**
   - BRD_ingestion.py (To extract BRDs)
   - agreement_ingestion.py (To extract agreement approvals)

## To-Do
   - need a better tool/library to extract data from tables inside PDFs
   - better chunking technique?
