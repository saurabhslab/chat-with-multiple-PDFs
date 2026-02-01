# 📑 Chat with Multiple PDFs

A Streamlit application that lets you **upload multiple PDF documents and chat with them using Google Gemini models**.
The app extracts text from PDFs, splits it into chunks, embeds it using **Google Generative AI embeddings**, stores it in a **FAISS vector database**, and answers user questions using **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

* Upload **multiple PDF files**
* Ask natural-language questions about the documents
* Context-aware answers using **Gemini 2.5 Flash**
* Fast semantic search with **FAISS**
* Maintains chat history during a session
* Simple and clean **Streamlit chat UI**

---

## 🧠 How It Works

1. **PDF Upload**

   * PDFs are uploaded via the Streamlit sidebar.

2. **Text Extraction**

   * Text is extracted from each page using `PyPDF2`.

3. **Chunking**

   * The text is split into overlapping chunks for better retrieval.

4. **Embeddings**

   * Each chunk is converted into embeddings using:

     ```
     models/gemini-embedding-001
     ```

5. **Vector Store**

   * Embeddings are stored in a FAISS vector database.

6. **Question Answering**

   * User questions retrieve the most relevant chunks.
   * The LLM answers using **only the retrieved context**.

---

## 🛠 Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Google Generative AI (Gemini)**
* **FAISS**
* **PyPDF2**
* **python-dotenv**

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> ⚠️ Make sure your API key has access to **Google Generative AI (Gemini)**.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit app
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 💡 Usage

1. Upload one or more PDF files from the sidebar
2. Click **Process**
3. Ask questions in the chat input
4. Get answers based **only on the uploaded documents**

---

## ⚠️ Limitations

* Answers are limited to the content found in uploaded PDFs
* Large PDFs may take longer to process
* No persistent storage (session resets on refresh)

---
