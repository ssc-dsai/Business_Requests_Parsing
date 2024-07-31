# Business Requests Parsing

## Overview
This repository contains the tool to parse business requirements document and agreement aprovals under a directory of business requests.
The tool is essentially a pipeline that processes data inside PDFs and Excel Files, applys embeddings, and stores them in a Qdrant Databse.
The business requirements document pipeline also generate a summary for each business request and store them in a seperate vector database.

## Installing Dependencies (Developed Under Python 3.11.4)
   ```
   pip install -r requirements.txt
   ```

## Usage
```
python3 file_name_of_tool path_of_BR_directory BR_summary_collection_name BR_details_collection_name
```

![alt text](https://github.com/iy2004/pipelines/blob/main/example.png?raw=true)
A directory of business requests should look similiar to the image above
Passing in different command line arguments allows different functionalities

## Example Usage
```
python3 BRD_ingestion.py ./Output BR_summaries BR_Data
```

This command parses all the Business Requirements Document under the Output directory and stores it in a Qdrant collection named "BR_Summaries" and "BR_Data"

## Command Line Options
1. **Available Tools**
   - BRD_ingestion.py (To extract BRDs)
   - agreement_ingestion.py (To extract agreement approvals)

## Notes
   - agreement_ingestion.py contains the code to parse agreement approvals
   - BRD_ingestion.py contains the code to parse BRDs and generate summaries based of these BRDs
   - utils.py contains common utility functions shared by some of files

## To-Do
   - need a better tool/library to extract data from tables inside PDFs
   - better chunking technique?
