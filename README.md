# AI Chatbot with RAG for PDF Querying

This project implements an AI chatbot that utilizes Retrieval Augmented Generation (RAG) to answer questions based on the content of PDF documents.

## Dependencies

* Streamlit
* LangChain
* Groq LLM
* Google Embeddings
* FAISS
* PyPDFDirectoryLoader
* RecursiveCharacterTextSplitter
* dotenv

To install the dependencies, run:

```bash
pip install -r requirements.txt
```
## Getting Started

1. **API keys:** Add your API keys in ragchatbot.py
2. **Add Your PDFs:** Place your PDF files into the `pdf` folder within the project directory (a sample pdf is present in the folder).
3.  **Run the Chatbot:** Open your terminal and navigate to the project's root directory. Execute the following command:

    ```bash
    streamlit run ragchatbot.py [ARGUMENTS]
    ```

    * `[ARGUMENTS]` are optional arguments you can pass to the streamlit application. For example, you can specify a port using `--server.port 8502`.

4.  **Access the Chatbot:** Once the command is executed, Streamlit will provide a local URL in the terminal. Open that URL in your web browser to access the chatbot.
5.  **Query the pdf:** Ask questions to the RAG agent which will answer based on the contents of the pdf.


