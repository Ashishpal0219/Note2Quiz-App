# Note2Quiz: AI-Powered Question Generator 📝🧠

This Streamlit application transforms lecture notes or other text documents (DOCX/PDF) into various types of quiz questions (MCQ, Short Answer, Viva) using the Google Gemini API, complete with Bloom's Taxonomy levels.

## Features ✨
* Upload DOCX or PDF files, or paste text directly.
* Generates MCQ, Short Answer, and Viva questions based on content.
* Assigns Bloom's Taxonomy levels (Remembering, Understanding, etc.) with reasoning.
* Exports quizzes to DOCX and CSV formats (solved and unsolved versions).
* Stores generated questions in a local SQLite database (Question Bank).
* Filter and view the Question Bank within the app.

## Tech Stack 🛠️
* **Python**
* **Streamlit** (Web Interface)
* **Google Gemini API** (AI Generation via `google-generativeai`)
* **spaCy** (Text Summarization, Topic Extraction)
* **Pandas** (Data Handling, CSV Export)
* **python-docx** (DOCX Reading/Writing)
* **PyPDF2** (PDF Reading)
* **SQLite** (Database)
* **python-dotenv** (API Key Management)

## Setup ⚙️
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ashishpal0219/Note2Quiz-App.git](https://github.com/Ashishpal0219/Note2Quiz-App.git)
    cd Note2Quiz-App
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Activate (Windows PowerShell)
    .venv\Scripts\Activate.ps1
    # Activate (Mac/Linux)
    # source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
4.  **Create `.env` file:** Create a file named `.env` in the project root and add your Google API key:
    ```env
    GOOGLE_API_KEY=YourActualGoogleGeminiApiKeyGoesHere
    ```
5.  **Create necessary folders:**
    ```bash
    mkdir database exports styles
    ```
    *(Note: `styles/custom.css` is optional)*

## Running the App 🚀
```bash
streamlit run app.py