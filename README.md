# 🎧 Audio Embeddings with Azure Cognitive Search

<img src="embedding.png" alt="Audio embedding illustration">

**Vector embeddings** are a way of representing content such as text, images, or audio as vectors of real numbers in a high-dimensional space. These embeddings are often learned from large amounts of data and can be used to measure semantic similarity between pieces of content.

Azure Cognitive Search currently doesn’t provide a built‑in way to vectorize documents and queries, leaving it up to you to select and run the best embedding model for your data.

In this project, we use **PANNS (Large-Scale Pretrained Audio Neural Networks)** to generate **audio embeddings** and store them in **Azure Cognitive Search**’s vector store to enable similarity search between audio files. 🔍🎵

We can then use these audio embeddings to:

- Find similarities between audio files
- Detect anomalies in sound
- Build intelligent audio search and monitoring scenarios

<img src="acs1.png" alt="Azure Cognitive Search vector search">

---

## 🧱 Project Structure

- 📘 [Audio Search with Azure Cognitive Search notebook](Audio%20Search%20with%20audio%20embeddings%20and%20Azure%20Cognitive%20Search.ipynb)  
  End‑to‑end example of audio similarity search using embeddings + Azure Cognitive Search.

- 📙 [Audio anomalies detection notebook](Audio%20anomalies%20detection.ipynb)  
  Uses audio embeddings to detect anomalous sounds (e.g., unexpected events in an audio stream).

- 🎵 `audio/`  
  Sample audio files used by the notebooks.

- 🖼️ `embedding.png`, `acs1.png`, `SED.png`  
  Illustrations for embeddings, Azure Cognitive Search, and sound event detection concepts.

---

## 🛠️ End-to-End Process

This repo demonstrates a typical **audio-embedding + vector search** workflow:

1. **Prepare audio data** 🎼  
   - Collect a catalog of audio files (e.g., `.wav`, `.mp3`) and place them under the `audio/` folder or another accessible location.
   - Optionally normalize or resample audio to a consistent sample rate.

2. **Generate audio embeddings with PANNS** 🧠  
   - Load a pretrained PANNS model (e.g., a CNN model trained on AudioSet).
   - For each audio file:
     - Load the waveform with a library such as `librosa` or `torchaudio`.
     - Convert it into the input format expected by the PANNS model.
     - Run a forward pass through the model to obtain a **fixed-length embedding vector** (e.g., 512 or 2048 dimensions).
   - Store the embeddings together with metadata (file name, label, etc.) in a structured format (e.g., Pandas DataFrame or JSON).

3. **Create an Azure Cognitive Search index with vector fields** ☁️  
   - Define an index schema that includes:
     - A key field (e.g., `id`)
     - Metadata fields (e.g., `fileName`, `label`, `duration`)
     - A **vector field** (e.g., `audioVector`) with:
       - `dimensions` = embedding size
       - `vectorSearchAlgorithm` (e.g., HNSW)
   - Provision the index in Azure Cognitive Search.

4. **Upload embeddings to Azure Cognitive Search** ⬆️  
   - Convert your embeddings into documents compatible with your index schema.
   - Use the Azure SDK for Python (e.g., `azure-search-documents`) to:
     - Connect to the search service
     - Upload (index) documents containing both metadata and the embedding vector.

5. **Perform similarity search using an audio query** 🔍  
   - Take a **query audio file**, generate its embedding using the **same PANNS model**.
   - Call Azure Cognitive Search with a **vector query** on the embedding field, retrieving the `k` nearest neighbors.
   - Inspect the results: similar audio clips, similarity scores, and associated metadata.

6. **(Optional) Anomaly detection** 🚨  
   - Learn the “normal” distribution of embeddings for healthy or expected sounds.
   - For a new audio embedding:
     - Compute its distance to the nearest neighbors or to the cluster center of normal data.
     - If the distance exceeds a threshold, mark it as **anomalous**.
   - Use this for monitoring use cases (machines, environments, sensors, etc.).

---

## 🐍 Python & Notebook Logic Overview

The Python code in the notebooks typically follows this structure:

### 1. Environment & Dependencies

The notebooks use common Python libraries such as:

- `numpy`, `pandas` – data manipulation
- `librosa` or `torchaudio` – audio loading and preprocessing
- `torch` – running the PANNS model (if using the PyTorch implementation)
- `azure-search-documents` – interacting with Azure Cognitive Search
- Plotting libraries for inspecting signals or embeddings (e.g., `matplotlib`)

You’ll usually see cells that:

- Install missing libraries (for hosted environments)
- Import all required modules
- Configure environment variables or secrets (Search service name, key, index name, etc.)

### 2. Loading and Processing Audio

Typical audio processing steps in Python are:

```python
import librosa
import numpy as np

file_path = "audio/example.wav"
waveform, sr = librosa.load(file_path, sr=32000, mono=True)  # resample to 32 kHz

# Optional: trim silence, normalize, or pad/clamp to a fixed duration
```

The notebooks then format audio into the tensor shape expected by the PANNS model (e.g., `[batch, time]` or `[batch, channel, time]`).

### 3. Generating Embeddings with PANNS

The PANNS model is usually loaded as a pretrained network, for example:

```python
import torch

# Pseudocode – exact class and weights path depend on the implementation in the notebook
model = PannsCNN(pretrained=True)
model.eval()

with torch.no_grad():
    # Assume `audio_tensor` is [batch, time] or [batch, channel, time]
    embedding = model(audio_tensor)
    # embedding: [batch, embedding_dim]
```

The resulting `embedding` tensor is then converted to a NumPy array or Python list:

```python
embedding_vector = embedding.squeeze(0).cpu().numpy().tolist()
```

These vectors are later stored and sent to Azure Cognitive Search.

### 4. Building the DataFrame / Document List

The notebooks typically construct a collection like:

```python
import pandas as pd

records = []

for file_path in audio_files:
    # 1. Load audio
    # 2. Compute embedding_vector
    records.append({
        "id": some_unique_id,
        "fileName": file_path,
        "audioVector": embedding_vector,
        # optional metadata...
    })

df = pd.DataFrame(records)
```

This DataFrame is a convenient intermediate step before pushing data to the search index.

### 5. Creating the Azure Cognitive Search Index

Using `azure-search-documents`, the Python code:

- Authenticates using the service endpoint and admin key
- Defines the index schema, including the vector field

Example (simplified conceptual structure):

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    SearchField
)
from azure.core.credentials import AzureKeyCredential

endpoint = "https://<your-service-name>.search.windows.net"
admin_key = "<your-admin-key>"
index_name = "audio-embeddings-index"

credential = AzureKeyCredential(admin_key)
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="fileName", type=SearchFieldDataType.String, filterable=True, searchable=True),
    SearchField(
        name="audioVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=EMBEDDING_DIM,  # e.g., 1024
        vector_search_configuration="audio-vector-config",
    ),
]

vector_search = VectorSearch(
    algorithm_configurations=[
        HnswVectorSearchAlgorithmConfiguration(
            name="audio-vector-config",
            kind="hnsw"
        )
    ]
)

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
)

index_client.create_index(index)
```

The exact field names and dimensions are defined in the notebook; the above is representative.

### 6. Uploading Embeddings as Documents

Once the index exists, the notebook uses a `SearchClient` to upload documents:

```python
from azure.search.documents import SearchClient

search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=credential
)

documents = df.to_dict(orient="records")
result = search_client.upload_documents(documents=documents)
```

Each document includes:

- `id` – unique identifier
- `fileName` – path or human‑readable name
- `audioVector` – embedding list of floats
- Any other metadata fields you configured

### 7. Running Vector Similarity Search

To search using an **audio query**, the notebook:

1. Loads the query audio file
2. Computes its embedding with the **same PANNS model**
3. Issues a vector search request against the `audioVector` field

Example (pseudocode):

```python
query_embedding = get_embedding("audio/query.wav")  # same as for catalog items

results = search_client.search(
    search_text="",  # empty for pure vector search
    vectors=[
        {
            "value": query_embedding,
            "fields": "audioVector",
            "k": 5,  # top 5 most similar
        }
    ]
)

for result in results:
    print(result["fileName"], result["@search.score"])
```

---

## 💼 Example Business Applications

- 🧑‍🤝‍🧑 **Gender detection** from voice
- 🙂 **Sentiment analysis** on spoken audio
- 🛠️ **Predictive maintenance** (e.g., machinery / equipment sounds)
- ⚠️ **Anomaly detection** (unusual events, alarms, abnormal behavior)

You can adapt the notebooks to your own domain by changing:

- The audio dataset in `audio/`
- The index schema
- The post‑processing / decision logic (e.g., thresholds for anomalies)

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/retkowsky/audio-embeddings.git
   cd audio-embeddings
   ```

2. Open the notebooks in Jupyter / VS Code / Azure ML:
   - `Audio Search with audio embeddings and Azure Cognitive Search.ipynb`
   - `Audio anomalies detection.ipynb`

3. Configure your Azure Cognitive Search service:
   - Set environment variables or directly paste:
     - Service endpoint
     - Admin API key
     - Index name

4. Run the notebooks cell by cell to:
   - Generate embeddings
   - Create the index
   - Upload documents
   - Perform similarity or anomaly detection queries

---

## 📚 Learn More

- [What is Azure Cognitive Search?](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)  
- [Announcing vector search in Azure Cognitive Search (public preview)](https://techcommunity.microsoft.com/t5/azure-ai-services-blog/announcing-vector-search-in-azure-cognitive-search-public/ba-p/3872868)

---

<br>

📅 **Last updated:** 17 July 2023  

👤 **Author:** Serge Retkowsky  
📧 serge.retkowsky@microsoft.com  
🔗 [LinkedIn – Serge Retkowsky](https://www.linkedin.com/in/serger/)