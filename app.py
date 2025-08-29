# app.py
import os
import json
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# 0) Config
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("⚠️ Missing GOOGLE_API_KEY in .env file")
INDEX_PATH = os.environ.get("INDEX_PATH", "faiss_index")

# 1) Load FAISS index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if not os.path.exists(INDEX_PATH):
    raise RuntimeError(f"FAISS index path not found: {INDEX_PATH}")
vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# 2) Setup QA chain (Gemini)
prompt_template = """
You are a specialist in Mental Health and Substance Use Disorders in Africa.
Answer strictly based on the provided data. Do not invent information.
Identify the exact variable (indicator) and/or the dataset that corresponds to the user’s question but don't mention it.
Answer clearly and politely but be short and accurate.
Always finish politely by asking:
"Would you like to explore further insights from the Mental Health Dashboard?"

Context: {context}
Question: {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 3) API function
def ask_api(question: str, k: int = 20, filters_json: str = ""):
    try:
        k = int(k)
    except:
        k = 20
    docs = vector_store.similarity_search(question, k=k)
    used_filters = {}
    if filters_json:
        try:
            used_filters = json.loads(filters_json)
            filtered = []
            for d in docs:
                keep = True
                for key, val in used_filters.items():
                    if f"{key}: {val}" not in d.page_content:
                        keep = False
                        break
                if keep:
                    filtered.append(d)
            docs = filtered
        except Exception as e:
            used_filters = {"_error": f"Invalid JSON: {e}"}
    response = qa_chain.invoke({"input_documents": docs, "question": question})
    answer = response.get("output_text", "").strip()
    return {"answer": answer, "num_docs": len(docs), "used_filters": used_filters}

# 4) Gradio UI + API
with gr.Blocks(title="Mental Health RAG API") as demo:
    gr.Markdown("## Mental Health Dashboard — RAG API (FAISS + Gemini)")
    with gr.Row():
        question = gr.Textbox(label="Question", lines=2, placeholder="Ask about mental health indicators...")
    with gr.Row():
        k = gr.Slider(1, 50, value=20, step=1, label="Top-K")
        filters_json = gr.Textbox(label="Filters (JSON, optional)", placeholder='{"Country":"Algeria","Year":"2020"}')
    out = gr.JSON(label="Response")
    btn = gr.Button("Ask")
    btn.click(ask_api, inputs=[question, k, filters_json], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
