# DeepRAGHub

DeepRAGHub is an advanced Retrieval-Augmented Generation (RAG) system that combines the power of large language models with efficient document retrieval for enhanced question-answering capabilities.

## 🌟 Features

- 📚 Flexible document loading and preprocessing
- 🧠 Customizable embedding models
- 🚀 Efficient vector storage and retrieval using Qdrant
- 🤖 Support for various LLM backends (OpenAI, Hugging Face)
- 🔧 Configurable RAG pipeline

## 📁 Project Structure
```bash
DeepRAGHub/
├── data/
│ ├── raw/
│ └── processed/
├── deepraghub/
│ ├── data/
│ ├── embedding/
│ ├── generation/
│ ├── retrieval/
│ ├── rag/
│ └── utils/
│ └── config/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```
<br>

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DeepRAGHub.git
   cd DeepRAGHub
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
<br>

## ⚙️ Configuration

The system is highly configurable. Modify the `deepraghub/utils/config/config.yaml` file to adjust settings for:

- Data processing
- Embedding model
- Vector storage
- LLM generation
- RAG pipeline

For detailed configuration options, refer to the `config.yaml` file in the repository.
<br>

## 🚀 Usage

1. Prepare your documents by placing them in the `data/raw` directory.

2. Run the main script:
   ```bash
   python main.py
   ```

This script will:
- Load and preprocess documents
- Initialize and (optionally) train the embedding model
- Embed documents and store them in the vector database
- Set up the LLM and RAG pipeline
- Run a sample query

For custom usage, you can import the necessary components:

<br>

```python
from deepraghub import load_config, load_documents, chunk_documents, EmbeddingModel, VectorStore, LLMModel, RAGPipeline

### Load configuration
config = load_config("path/to/your/config.yaml")
### Create RAG pipeline
rag_pipeline = RAGPipeline(llm_model, vector_store, max_context_size=config.rag.max_context_size, top_k_docs=config.rag.top_k_docs)
### Query
answer = rag_pipeline.query("Your question here")
print(answer)
```

<br>

## 🏗️ Architecture

DeepRAGHub consists of several key components:

1. **Data Processing**: Handles document loading and chunking.
2. **Embedding**: Converts text chunks into vector representations.
3. **Vector Store**: Efficiently stores and retrieves document embeddings.
4. **LLM Integration**: Interfaces with various language models for text generation.
5. **RAG Pipeline**: Orchestrates the retrieval and generation process.

<br>

## 🔧 Extending the System

- To add new embedding models, extend the `EmbeddingModel` class in `deepraghub/embedding/model.py`.
- For new LLM backends, modify the `LLMFactory` in `deepraghub/generation/llm.py`.
- Custom document loaders can be added to `deepraghub/data/loader.py`.