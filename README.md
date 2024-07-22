# Business Requests Parsing

## Overview
This repository contains the tool to parse business requirements document and agreement aprovals under a business request directory.
The tool is essentially a pipeline that processes data inside PDFs and Excel Files, applys embeddings, and stores them in a Qdrant Databse.
The business requirements document pipeline also uses a classification model to apply labels to all the images extracted from an Excel Worksheet.

## Installing Dependencies (Developed Under Python 3.11.4)
   ```
   pip install -r requirements.txt
   ```

## Usage
```
python3 "file_name_of_tool" "path_of_BR_directory" "collection_name"  
```

Passing in different command line arguments allows different functionalities

## Command Line Options
1. **Available Tools**
   - BRD_ingestion.py (To extract BRDs)
   - agreement_ingestion.py (To extract agreement approvals)
2. **Available Collections**
   - BR_Json

## Notes
   - images folder contains a garbage collection produced while doing image labeling (will find a way to fix this)
   - model folder contains the code to train and create a model that classifies labels inside Excel BRDs
   - agreement_ingestion.py contains the code to parse agreement approvals
   - BRD_ingestion.py contains the code to parse BRDs
   - utils.py contains common utility functions shared by some of files

## To-Do
   - need a better tool/library to extract data from tables inside PDFs
   - better chunking technique?
   - not 100% accurate (number in section 3.14 of the Computing worksheet in BRD of BR70067 cannot be detected)
