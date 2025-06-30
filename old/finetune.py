import os
import re
import json
import requests
import numpy as np
import psycopg2
from supabase import create_client
from dotenv import load_dotenv

# --- Set your database credentials here or in a .env file ---
os.environ["user"] = "postgres.ojbalezgbnwunzzoajum"
os.environ["password"] = "wheelychatbot"
os.environ["host"] = "aws-0-ap-south-1.pooler.supabase.com"
os.environ["port"] = "5432"
os.environ["dbname"] = "postgres"

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
supabase_url = "https://ojbalezgbnwunzzoajum.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9qYmFsZXpnYm53dW56em9hanVtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0OTk0MzMsImV4cCI6MjA2NTA3NTQzM30.20UaD3p7f1PHDCUhyEO4n3orWGqB-ku7pzBQLESXh4E"
supabase = create_client(supabase_url, supabase_key)

# --- Azure OpenAI Chat Endpoint and Headers ---
chat_endpoint = "https://ai-api-dev.dentsu.com/openai/deployments/GPT35Turbo/chat/completions?api-version=2024-10-21"
chat_headers  = {
    "x-service-line": "functions",
    "x-brand": "dentsu",
    "x-project": "aiassistant",
    "api-version": "v15",
    "Ocp-Apim-Subscription-Key": "3c6489668e324e6e8123e94f41456484"
}

def rewrite_query_for_rag(user_query):
    """
    Rewrites user query to make it neutral and more suitable for vector similarity search.
    Removes conversational elements, expands abbreviations, and focuses on key concepts.
    """
    system_prompt = (
        "You are an expert at rewriting queries for better vector similarity search in a tyre manufacturing context. "
        "Your task is to rewrite user queries to make them more neutral, comprehensive, and suitable for semantic search. "
        "\n\nGuidelines for rewriting:\n"
        "1. Remove conversational elements (please, can you, I want to know, etc.)\n"
        "2. Expand abbreviations and acronyms related to tyres (e.g., 'R' to 'radial')\n"
        "3. Add relevant synonyms and related terms\n"
        "4. Convert questions to declarative statements focusing on key concepts\n"
        "5. Include industry-specific terminology where appropriate\n"
        "6. Maintain all specific identifiers (IDs, part numbers, names)\n"
        "7. Focus on the core information need\n"
        "\nExamples:\n"
        "Original: 'Can you tell me the status of claim 10010?'\n"
        "Rewritten: 'claim status warranty service claim ID 10010 processing pending approved rejected'\n"
        "\nOriginal: 'What tyres do we have in Mumbai warehouse?'\n"
        "Rewritten: 'tyre inventory stock products Mumbai warehouse location available quantity'\n"
        "\nOriginal: 'Show me sales data for dealer Sunita Tyres'\n"
        "Rewritten: 'sales data transactions dealer Sunita Tyres revenue quantity sold products performance'\n"
        "\nOriginal: 'What's the price of 175/65R14 tyres?'\n"
        "Rewritten: 'price cost 175/65R14 tyre product pricing section width aspect ratio radial construction'\n"
        "\nOnly return the rewritten query, nothing else."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    payload = {
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 150,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    try:
        response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
        if response.status_code == 200:
            rewritten_query = response.json()["choices"][0]["message"]["content"].strip()
            print(f"DEBUG: Original query: {user_query}")
            print(f"DEBUG: Rewritten query: {rewritten_query}")
            return rewritten_query
        else:
            print(f"Query rewriting failed: {response.status_code}: {response.text}")
            return user_query  # Fall back to original query
    except Exception as e:
        print(f"Query rewriting error: {e}")
        return user_query  # Fall back to original query

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

def similarity_search(query_embedding, rows, top_k=3):
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
    rows_as_strings = []
    for row in sql_result:
        row_str = ', '.join(f"{k}: {v}" for k, v in row.items())
        rows_as_strings.append(row_str)
    return "\n---\n".join(rows_as_strings)

def get_llm_sql(user_query):
    system_prompt = (
        "You are an AI assistant for a tyre manufacturing company. "
        "You help generate efficient SQL SELECT queries from user questions based on the following PostgreSQL schema.\n\n"
        "Tables and their descriptions:\n"
        "1. users(user_id, username, password, email, role, dealer_id): Stores user login and profile information, including their role and associated dealer.\n"
        "2. dealer(dealer_id, name): Stores information about tyre dealers.\n"
        "3. claim(claim_id, dealer_id, status): Stores warranty or service claims made by dealers.\n"
        "4. product(product_id, category, price, section_width, aspect_ratio, construction_type, rim_diameter_inch): Stores product (tyre) details, including size and pricing.\n"
        "5. warehouse(warehouse_id, location, zone): Stores warehouse locations and zones.\n"
        "6. sales(sales_id, dealer_id, product_id, warehouse_id, quantity, cost, date): Stores sales transactions, including which dealer sold which product from which warehouse, quantity, cost, and date.\n"
        "7. inventory(product_id, warehouse_id, quantity): Stores current stock levels of each product in each warehouse.\n\n"
        "Key relationships:\n"
        "- users joins dealer on dealer_id\n"
        "- claim joins dealer on dealer_id\n"
        "- sales joins dealer, product, and warehouse via dealer_id, product_id, and warehouse_id\n"
        "- inventory joins product and warehouse via product_id and warehouse_id\n\n"
        "Rules for Generating SQL:\n"
        "1. Use ONLY the columns and tables exactly as defined above. Do NOT invent or assume any columns or tables that are not listed.\n"
        "2. Before writing a JOIN, double-check that both tables and columns exist in the schema above.\n"
        "3. If a table or column is not present in the schema above, do NOT use it in the SQL.\n"
        "4. Identify relevant tables by checking which entities are mentioned (e.g., product, dealer, warehouse, SKU, claim, sales).\n"
        "5. Use INNER JOINs only when both tables are required for filtering or output.\n"
        "6. Use WHERE clauses to filter by SKU, warehouse location, claim status, dealer name, product category, date, or role.\n"
        "7. Use GROUP BY and aggregation (SUM, COUNT, MAX) when asked about totals, trends, or summaries.\n"
        "8. Use meaningful aliases and readable column names in SELECT.\n"
        "9. Use dealer_id or user role to restrict results for dealer-specific queries.\n"
        "10. If a user query mentions a tyre size like '110/80R17 57H', treat it as a product_id string and use it directly in WHERE clauses (e.g., WHERE product_id = '110/80R17 57H'). Do NOT attempt to parse or map the product_id to section_width, aspect_ratio, construction_type, or rim_diameter_inch. These columns are independent of product_id.\n"
        "11. Always SELECT and return all columns that are necessary to directly answer the user's question. For example, if the user asks about products in a specific warehouse, include both product and warehouse information in the SELECT statement. Do not omit relevant columns.\n"
        "12. For string comparisons (e.g., names, locations, SKUs), use case-insensitive matching by using ILIKE or LOWER() as appropriate.\n\n"
        "Only generate SQL SELECT statements. Do not include INSERT, UPDATE, or DELETE.\n"
        "If the question is vague, ambiguous, or cannot be answered with the schema, respond with 'NO_SQL'.\n"
        "Never use a table or column unless it is present in the schema above. If unsure, respond with 'NO_SQL'.\n\n"
        "Example questions and responses:\n"
        "Q: What is the status of claim ID 10010?\n"
        "A: SELECT status FROM claim WHERE claim_id = '10010';\n\n"
        "Q: How many tyres were sold by Dealer X last month?\n"
        "A: SELECT SUM(s.quantity) FROM sales s \n"
        "JOIN dealer d ON s.dealer_id = d.dealer_id \n"
        "WHERE d.name ILIKE 'Dealer X' \n"
        "AND s.date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') \n"
        "AND s.date < DATE_TRUNC('month', CURRENT_DATE);\n\n"
        "Q: Show inventory of product 110/80R17 57H in all warehouses.\n"
        "A: SELECT w.location, i.quantity \n"
        "FROM inventory i \n"
        "JOIN warehouse w ON i.warehouse_id = w.warehouse_id \n"
        "WHERE i.product_id = '110/80R17 57H';\n\n"
        "Q: List all claims made by dealer 'Sunita Tyres'.\n"
        "A: SELECT c.claim_id, c.status \n"
        "FROM claim c \n"
        "JOIN dealer d ON c.dealer_id = d.dealer_id \n"
        "WHERE d.name ILIKE 'Sunita Tyres';\n\n"
        "Q: What products are available in the Mumbai warehouse?\n"
        "A: SELECT p.product_id, p.category, i.quantity \n"
        "FROM inventory i \n"
        "JOIN product p ON i.product_id = p.product_id \n"
        "JOIN warehouse w ON i.warehouse_id = w.warehouse_id \n"
        "WHERE w.location ILIKE 'Mumbai';\n\n"
        "Q: What is the email of user 'Sunita Verma'?\n"
        "A: SELECT email FROM users WHERE username ILIKE 'Sunita.Verma';\n\n"
        "Q: What is the total stock of product 175/65R14 82T across all warehouses?\n"
        "A: SELECT SUM(quantity) AS total_stock FROM inventory WHERE product_id = '175/65R14 82T';\n\n"
        "Your task: Given any valid user query, generate the most efficient SQL SELECT statement using the schema above. "
        "If not possible, respond with 'NO_SQL'. "
        "Never use a column or table not present in the schema above."
        
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
            dbname=os.getenv("dbname"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port")
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
        "Your name is 'shivam'. "
        "Always use the provided context from the database to answer the user's query. "
        "If the context contains relevant information, present it clearly in your answer. "
        "If the context is a table or list of key-value pairs, restate the information directly as the answer. "
        "If the context contains a single value (e.g., 'user_id: 3'), return that value as the answer. "
        "If the context contains one or more rows like 'claim_id: 10001, status: Pending', present these rows as the answer. "
        "If the context is empty or says 'No results found.', say so. "
        "Do not invent data or refer to information not present in the context. "
        "Be concise and direct in your answer."
        "\n\n"
        "Example:\n"
        "Context:\nclaim_id: 10001, status: Pending\n\nQuery: whats the claim status of dealer id 3\n"
        "Answer: claim_id: 10001, status: Pending"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_query}"}
    ]
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Chat API error: {response.status_code}: {response.text}")

def vector_store_similarity_search(query_embedding, top_k=10, metadata_filter=None, similarity_threshold=0.1):
    """
    Enhanced vector similarity search with similarity threshold filtering.
    """
    response = supabase.table("vector_store").select("*").execute()
    rows = response.data
    if not rows:
        return []
    
    # Metadata filtering: keep rows with the most metadata matches
    if metadata_filter:
        match_counts = []
        for row in rows:
            try:
                meta = row.get("metadata")
                if isinstance(meta, str):
                    meta = json.loads(meta)
                # Count matching keys (case-insensitive)
                match_count = sum(
                    str(meta.get(k, "")).lower() == str(v).lower()
                    for k, v in metadata_filter.items()
                )
                match_counts.append((match_count, row))
            except Exception:
                continue
        if match_counts:
            max_count = max(count for count, _ in match_counts)
            # Only keep rows with the maximum match count and at least one match
            rows = [row for count, row in match_counts if count == max_count and count > 0]
        else:
            rows = []
    
    # Vector similarity with threshold filtering
    similarities = []
    for row in rows:
        try:
            row_embedding = json.loads(row["embedding"])
            sim = cosine_similarity(query_embedding, row_embedding)
            # Only include results above similarity threshold
            if sim >= similarity_threshold:
                similarities.append((sim, row))
        except Exception:
            continue
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    results = [row for sim, row in similarities[:top_k]]
    
    # Debug information
    if similarities:
        print(f"DEBUG: Found {len(similarities)} results above threshold {similarity_threshold}")
        print(f"DEBUG: Top similarity scores: {[round(sim, 3) for sim, _ in similarities[:5]]}")
    
    return results

def vector_rows_to_context(rows):
    context = ""
    for idx, row in enumerate(rows, 1):
        context += f"\nVector Row {idx}:\n"
        for key, value in row.items():
            if key != "embedding":
                context += f"{key}: {value}\n"
    return context.strip()

def get_llm_final_response(sql_context, rag_context, user_query):
    system_prompt = (
        "You are an AI assistant for the tyre manufacturing industry. act like a assistant. "
        "Your name is 'shivam'. "
        "You are given two sources of information: "
        "1. SQL database results (structured, factual data) "
        "2. Retrieved knowledge (RAG) from a vector store (unstructured, descriptive data). "
        "Use both sources to answer the user's query as accurately as possible. "
        "If both are relevant, combine the information. "
        "If only one is relevant, use that. "
        "If neither is relevant, say so. "
        "answer like a assistant to the user no need to let them know about the sources of information. act like they are talking to a human assistant. "
        "Do not invent data or refer to information not present in the context."
        "\n\n"
        "SQL Context:\n{sql_context}\n\n"
        "RAG Context:\n{rag_context}\n\n"
        "User Query: {user_query}"
    )
    messages = [
        {"role": "system", "content": system_prompt.format(
            sql_context=sql_context, rag_context=rag_context, user_query=user_query)},
        {"role": "user", "content": user_query}
    ]
    payload = {
        "messages": messages,
        "temperature": 0.0,
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

def extract_metadata_with_llm(user_query):
    """
    Uses LLM to extract metadata fields from the user query.
    Returns a dict or None.
    """
    system_prompt = (
        "You are an assistant that extracts structured metadata from user queries about tyres, warehouses, dealers, claims, sales, and inventory. "
        "Given a user query, return a JSON object with as many of these fields as possible if present: "
        "claim_id, dealer_id, dealer_name, status, product_id, product_name, category, warehouse_id, location, zone, quantity, sales_id, date, cost, product_price, warehouse_location. "
        "If a field is not present, omit it. Only return a valid JSON object, no explanation or extra text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 200,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        print("DEBUG: Raw LLM metadata output:", content)
        try:
            metadata = json.loads(content)
            if isinstance(metadata, dict):
                return metadata
        except Exception as e:
            print("DEBUG: Metadata JSON decode error:", e)
            return None
    return None

def main():
    """
    Enhanced main function with query rewriting for improved RAG performance.
    """
    print("Enhanced RAG System with Query Rewriting")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_query = input("query : ")
        if user_query.lower() == "exit":
            break
        
        try:
            # Step 1: SQL Generation and Execution (unchanged)
            sql = get_llm_sql(user_query)
            sql_context = "No results found."
            
            if sql.strip().upper() == "NO_SQL":
                print("SQL Generation: No SQL could be generated for this query.")
            elif sql.strip().lower().startswith("select"):
                sql_result, sql_error = try_select_sql(sql)
                if sql_result is not None:
                    print("SQL Generation: SQL generated and executed successfully.")
                    print(f"sql : {sql}")
                    sql_context = sql_result_to_context(sql_result)
                    print("SQL Result:")
                    print(sql_context)
                else:
                    print("SQL Generation: SQL was generated but could not be executed.")
                    print(f"sql : {sql}")
                    print(f"SQL Error: {sql_error}")
            else:
                print("SQL Generation: Only SELECT statements are allowed. No valid SQL generated.")

            # Step 2: Enhanced RAG pipeline with query rewriting
            print("\n" + "="*50)
            print("RAG Pipeline with Query Rewriting")
            print("="*50)
            
            # Rewrite query for better vector search
            rewritten_query = rewrite_query_for_rag(user_query)
            
            # Process the rewritten query
            preprocessed_query = preprocess_query(rewritten_query)
            query_embedding = get_embedding(preprocessed_query)
            
            # Extract metadata from original query (not rewritten)
            metadata_filter = extract_metadata_with_llm(user_query)
            print("DEBUG: metadata_filter =", metadata_filter)
            
            # Enhanced vector search with similarity threshold
            vector_rows = vector_store_similarity_search(
                query_embedding, 
                top_k=10, 
                metadata_filter=metadata_filter,
                similarity_threshold=0.1  # Adjustable threshold
            )
            
            rag_context = vector_rows_to_context(vector_rows) if vector_rows else "No relevant vector context found."
            
            if vector_rows:
                print("RAG Vector Search Result:")
                print(rag_context)
            else:
                print("RAG Vector Search: No relevant results found above similarity threshold.")

            # Step 3: Generate final response combining SQL and RAG results
            print("\n" + "="*50)
            print("Final Response Generation")
            print("="*50)
            
            answer = get_llm_final_response(sql_context, rag_context, user_query)
            print(f"shivam : {answer}")

        except Exception as e:
            print(f"shivam : Sorry, I can't assist with that.")
            print("DEBUG Exception:", e)
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()