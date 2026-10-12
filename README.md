## README

### Ford EcoSport Local AI Diagnostic Assistant

A local RAG-based AI assistant for the **2022 Ford EcoSport Titanium service manual**. It retrieves relevant diagnostic information, identifies DTCs, follows Ford Pinpoint Test procedures, and generates grounded troubleshooting responses using **Gemma 3 4B** locally through Ollama.

### Main Technologies
- Python 3.11
- ChromaDB
- SentenceTransformers
- Ollama
- Gemma 3 4B
- HTML processing
- Retrieval-Augmented Generation (RAG)

### Resources Used

The primary knowledge source for this project is the **2022 Ford EcoSport Titanium service manual from LEMON Manuals**. The downloaded manual contains approximately **26,000 HTML pages** covering diagnostic procedures, DTC charts, wiring information, pinpoint tests, component information, and repair procedures.

The manual was processed locally and indexed in ChromaDB so that the AI can retrieve relevant sections instead of relying only on the language model's general knowledge.

**Source:** LEMON Manuals — 2022 Ford EcoSport Titanium Service Manual

> The manual content is used as the project's reference/knowledge source. This project does not claim ownership of the original Ford service-manual content.
