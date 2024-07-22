## Business Requests Parsing

# Overview
This repository contains the tool to parse business requirements document and agreement aprovals under a business request directory.
The tool is essentially a pipeline that processes data inside PDF and Excel Files, applys embeddings, and stores them in a Qdrant Databse.
The business requirements document pipeline also uses a classification model to apply labels to all the images extracted from an Excel Worksheet.

# Installing Dependencies (Developed Under Python 3.11.4)
   ```
   pip install -r requirements.txt
   ```

# Usage
