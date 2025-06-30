import os
import re
import json
import requests
import numpy as np
import psycopg2
from supabase import create_client
from dotenv import load_dotenv

# --- Set your database credentials here or in a .env file ---
os.environ["PGUSER"] = "postgres.qcjevsckdzernrygxzqe"
os.environ["PGPASSWORD"] = "Snapsupabase@20"
os.environ["PGHOST"] = "aws-0-ap-southeast-1.pooler.supabase.com"
os.environ["PGPORT"] = "6543"
os.environ["PGDATABASE"] = "postgres"

load_dotenv()

# --- Azure OpenAI Embedding Endpoint and Headers ---
embedding_endpoint = os.getenv("AZURE_OPENAI_URL")
embedding_headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY"),
    "x-service-line": os.getenv("AZURE_OPENAI_SERVICE_LINE"),
    "x-brand": os.getenv("AZURE_OPENAI_BRAND"),
    "x-project": os.getenv("AZURE_OPENAI_PROJECT"),
    'api-version': os.getenv("AZURE_OPENAI_API_VERSION")
}

# --- Supabase Client (for vector search) ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# --- Azure OpenAI Chat Endpoint and Headers ---
chat_endpoint = "https://ai-api-dev.dentsu.com/openai/deployments/GPT4o128k/chat/completions?api-version=2025-03-01-preview"
chat_headers  = {
    "x-service-line": "functions",
    "x-brand": "dentsu",
    "x-project": "aiassistant",
    "api-version": "v15",
    "Ocp-Apim-Subscription-Key": "43c73c423158406d825dfc2884a0bfea"
}

def preprocess_query(query):
    text = query.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_embedding(text):
    payload = {"input": text}
    response = requests.post(embedding_endpoint, headers=embedding_headers, json=payload)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Embedding API error {response.status_code}: {response.text}")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def similarity_search(query_embedding, rows, top_k=10):
    similarities = []
    for row in rows:
        try:
            row_embedding = json.loads(row["embedding"])
            sim = cosine_similarity(query_embedding, row_embedding)
            similarities.append((sim, row))
        except Exception:
            continue
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [row for sim, row in similarities[:top_k]]

def rows_to_context(rows):
    context = ""
    for idx, row in enumerate(rows, 1):
        context += f"\nRow {idx}:\n"
        for key, value in row.items():
            if key != "embedding":
                context += f"{key}: {value}\n"
    return context.strip()

def sql_result_to_context(sql_result):
    if not sql_result:
        return "No results found."
    context = ""
    for idx, row in enumerate(sql_result, 1):
        context += f"\nRow {idx}:\n"
        for key, value in row.items():
            context += f"{key}: {value}\n"
    return context.strip()

def get_llm_sql(user_query):
    system_prompt = (
        "You are an AI assistant for the manufacturing industry. "
        "There are three modules: Dealer, Admin, and All. "
        "Dealer can ask about SKU availability, their own sales, and claim statuses. "
        "Admin can ask about all data, system logs, and analytics. "
        "The main table is 'skus' with columns: id, name, category, zone, warehouse, stock, price, description, embedding. "
        "Given a user's question, generate a SQL SELECT statement to answer it using the skus table. "
        "If the question is insufficient or unclear to generate a SQL command, respond with 'NO_SQL'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
    if response.status_code == 200:
        sql = response.json()["choices"][0]["message"]["content"].strip()
        return sql
    else:
        raise Exception(f"Chat API error (SQL): {response.status_code}: {response.text}")

def try_select_sql(sql):
    sql = sql.strip()
    if not sql.lower().startswith("select"):
        return None, "Only SELECT statements are allowed for safety."
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT")
        )
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
        cur.close()
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

def get_llm_response(context, user_query):
    system_prompt = (
        "You are an AI assistant for the manufacturing industry. "
        "You must only answer queries that are directly related to the system, its data, or relevant general information about the manufacturing domain, products, SKUs, inventory, claims, or analytics. "
        "If a user asks anything unrelated to the system, its data, or the manufacturing domain, politely respond with: 'Sorry, I can only assist with queries related to this system and its data.' "
        "There are three modules: Dealer, Admin, and All. "
        "Dealer can ask about SKU availability, their own sales, and claim statuses. "
        "Admin can ask about all data, system logs, and analytics. "
        "Use the provided context from the database to answer the user's query as helpfully as possible. "
        "If similar products are provided in the context, mention only the top 3 as suggestions if appropriate."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_query}"}
    ]
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Chat API error: {response.status_code}: {response.text}")

def print_first_5_rows():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT")
        )
        cur = conn.cursor()
        cur.execute("SELECT * FROM skus LIMIT 5;")
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        print("First 5 rows from skus table:")
        for row in rows:
            print(dict(zip(columns, row)))
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching rows: {e}")

def main():
    while True:
        user_query = input("query : ")
        if user_query.lower() == "exit":
            break
        try:
            # 1. Generate SQL and run it (if possible)
            sql = get_llm_sql(user_query)
            sql_result, sql_error = None, None
            sql_context = ""
            if sql.strip().upper() != "NO_SQL" and sql.strip().lower().startswith("select"):
                sql_result, sql_error = try_select_sql(sql)
                sql_context = sql_result_to_context(sql_result) if sql_result is not None else ""
            else:
                sql = None

            # 2. Always run vector search (top 10)
            preprocessed_query = preprocess_query(user_query)
            query_embedding = get_embedding(preprocessed_query)
            response = supabase.table("skus").select("*").execute()
            rows = response.data
            vector_context = ""
            if rows:
                top_rows = similarity_search(query_embedding, rows, top_k=10)
                if top_rows:
                    vector_context = rows_to_context(top_rows)

            # 3. Combine both contexts for LLM
            combined_context = ""
            if sql and sql_context:
                combined_context += f"SQL Result:\n{sql_context}\n"
            if vector_context:
                combined_context += f"\nSimilar Products (Vector Search):\n{vector_context}"

            # 4. Feed to LLM for final answer
            answer = get_llm_response(combined_context.strip(), user_query)
            if sql:
                print(f"sql : {sql}")
            print(f"shivam : {answer}")
        except Exception as e:
            print(f"shivam : Sorry, I can't assist with that.")

if __name__ == "__main__":
   print_first_5_rows()
   main()
