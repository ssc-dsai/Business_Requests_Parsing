## Installation and Setup

1. **Installing Dependencies (Developed Under Python 3.11.4)**
   ```
   pip install -r requirements.txt
   ```
   
2. **Obtain an API Key from LLamaParse**

   ```
    https://cloud.llamaindex.ai/
   ```

3. **Set the environment variable to the API key**  
    LLAMA_PARSE_API_KEY="YOUR_API_KEY"

4. **Start Parsing the data using command line arguments**  
    Make sure your qdrant database is running
    ```
    python Json_Ingestion.py "YOUR_DIRECTORY_OF_BRs" "QDRANT_COLLECTION_NAME"
    ```
