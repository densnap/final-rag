import os
import re
import json
import requests
import numpy as np
import psycopg2
from supabase import create_client
from dotenv import load_dotenv
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process

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

# Cache for fuzzy matching
DEALER_CACHE = {}
PRODUCT_CACHE = {}
WAREHOUSE_CACHE = {}

def normalize_text(text):
    """Normalize text for better matching"""
    if not isinstance(text, str):
        text = str(text)
    # Convert to lowercase, remove extra spaces and special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def fuzzy_match_string(query_string, candidates, threshold=70):
    """
    Find the best fuzzy match for a string from a list of candidates
    Returns (best_match, score) or (None, 0) if no good match found
    """
    if not query_string or not candidates:
        return None, 0
    
    query_normalized = normalize_text(query_string)
    candidates_normalized = [normalize_text(c) for c in candidates]
    
    # Use fuzzywuzzy for better matching
    try:
        result = process.extractOne(query_normalized, candidates_normalized, scorer=fuzz.ratio)
        if result and result[1] >= threshold:
            # Find original candidate
            original_index = candidates_normalized.index(result[0])
            return candidates[original_index], result[1]
    except:
        pass
    
    return None, 0

def get_database_entities():
    """Cache database entities for fuzzy matching"""
    global DEALER_CACHE, PRODUCT_CACHE, WAREHOUSE_CACHE
    
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("dbname"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port")
        )
        cur = conn.cursor()
        
        # Get dealers
        cur.execute("SELECT DISTINCT name FROM dealer WHERE name IS NOT NULL")
        dealers = [row[0] for row in cur.fetchall()]
        DEALER_CACHE = {normalize_text(d): d for d in dealers}
        
        # Get products
        cur.execute("SELECT DISTINCT product_id FROM product WHERE product_id IS NOT NULL")
        products = [row[0] for row in cur.fetchall()]
        PRODUCT_CACHE = {normalize_text(p): p for p in products}
        
        # Get warehouses
        cur.execute("SELECT DISTINCT location FROM warehouse WHERE location IS NOT NULL")
        warehouses = [row[0] for row in cur.fetchall()]
        WAREHOUSE_CACHE = {normalize_text(w): w for w in warehouses}
        
        cur.close()
        conn.close()
        
        print(f"DEBUG: Cached {len(dealers)} dealers, {len(products)} products, {len(warehouses)} warehouses")
        
    except Exception as e:
        print(f"DEBUG: Error caching entities: {e}")

def fuzzy_correct_entities(text):
    """
    Find and correct entity names in the text using fuzzy matching
    """
    corrected_text = text
    corrections_made = []
    
    # Check for dealer names
    for cached_dealer in DEALER_CACHE.values():
        words_in_dealer = cached_dealer.lower().split()
        for word in words_in_dealer:
            if len(word) > 3:  # Only check meaningful words
                # Find potential matches in the text
                text_words = text.lower().split()
                for text_word in text_words:
                    similarity = fuzz.ratio(word, text_word)
                    if similarity >= 75 and word != text_word:
                        # Replace in original text (case sensitive)
                        pattern = re.compile(re.escape(text_word), re.IGNORECASE)
                        corrected_text = pattern.sub(word, corrected_text)
                        corrections_made.append(f"{text_word} -> {word}")
    
    # Check for product IDs (more strict matching)
    for cached_product in PRODUCT_CACHE.values():
        # For product IDs, use exact fuzzy matching
        if cached_product.lower() in text.lower():
            continue
        similarity = fuzz.ratio(normalize_text(cached_product), normalize_text(text))
        if similarity >= 80:
            corrected_text = corrected_text.replace(text, cached_product)
            corrections_made.append(f"Product: {text} -> {cached_product}")
    
    # Check for warehouse locations
    for cached_warehouse in WAREHOUSE_CACHE.values():
        similarity = fuzz.ratio(normalize_text(cached_warehouse), normalize_text(text))
        if similarity >= 80:
            pattern = re.compile(re.escape(text), re.IGNORECASE)
            corrected_text = pattern.sub(cached_warehouse, corrected_text)
            corrections_made.append(f"Warehouse: {text} -> {cached_warehouse}")
    
    if corrections_made:
        print(f"DEBUG: Fuzzy corrections made: {corrections_made}")
    
    return corrected_text

def rewrite_query_for_rag(user_query):
    """
    Enhanced query rewriting with fuzzy correction
    """
    # First, try to correct entity names
    corrected_query = fuzzy_correct_entities(user_query)
    
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
        "8. Include variations and synonyms for names and products to handle typos\n"
        "\nExamples:\n"
        "Original: 'Can you tell me the status of claim 10010?'\n"
        "Rewritten: 'claim status warranty service claim ID 10010 processing pending approved rejected'\n"
        "\nOriginal: 'What tyres do we have in Mumbai warehouse?'\n"
        "Rewritten: 'tyre inventory stock products Mumbai warehouse location available quantity'\n"
        "\nOriginal: 'Show me sales data for dealer Sunita Tyres'\n"
        "Rewritten: 'sales data transactions dealer Sunita Tyres revenue quantity sold products performance'\n"
        "\nOnly return the rewritten query, nothing else."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": corrected_query}
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
            print(f"DEBUG: Corrected query: {corrected_query}")
            print(f"DEBUG: Rewritten query: {rewritten_query}")
            return rewritten_query
        else:
            print(f"Query rewriting failed: {response.status_code}: {response.text}")
            return corrected_query
    except Exception as e:
        print(f"Query rewriting error: {e}")
        return corrected_query

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

#def similarity_search(query_embedding, rows, top_k=3):
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
    """Enhanced SQL generation with fuzzy entity correction"""
    # Correct entities in the query first
    corrected_query = fuzzy_correct_entities(user_query)
    
    system_prompt = (
    "You are an AI assistant for a tyre manufacturing company. "
    "You help generate efficient SQL SELECT queries from user questions about inventory stock and claim statuses, based on the following PostgreSQL schema.\n\n"
    
    "Relevant Tables and their descriptions:\n"
    "1. claim(claim_id, dealer_id, status, claim_date, resolved_date, product_id, amount, approved_amount, reason): "
    "Stores warranty or service claims made by dealers, including dates, product info, and financial details.\n"
    "2. product(product_id, product_name, category, price, section_width, aspect_ratio, construction_type, rim_diameter_inch): "
    "Stores product (tyre) details, including size and pricing.\n"
    "3. inventory(product_id, warehouse_id, quantity): "
    "Stores current stock levels of each product in each warehouse.\n"
    "4. warehouse(warehouse_id, location, zone): Stores warehouse locations and zones.\n"
    
    "Key relationships:\n"
    "- claim joins product on product_id\n"
    "- inventory joins product and warehouse via product_id and warehouse_id\n\n"
    
    "Rules for Generating SQL:\n"
    "1. Use ONLY the columns and tables exactly as defined above.\n"
    "2. For product names, locations, and zones, use ILIKE with % wildcards for partial and fuzzy matching.\n"
    "3. Always SELECT and return all columns necessary to answer the user's question.\n"
    "4. Use LOWER() for case-insensitive comparisons when needed.\n"
    "5. Use meaningful aliases and readable column names in SELECT.\n"
    "6. If a user query mentions a tyre size (like 175/65R14), treat it as a product_id and use ILIKE to match.\n"
    "7. Claims may be filtered by status (e.g., 'approved', 'pending'), date, or amount — support those filters.\n"
    "8. Inventory queries should return quantity available, and may include warehouse filters.\n"
    
    "Examples:\n"
    "SELECT i.quantity AS stock_quantity FROM product AS p JOIN inventory AS i ON p.product_id = i.product_id WHERE p.product_name ILIKE '%Agrotuff Cross%';"
    #"- SELECT i.quantity AS stock_quantity FROM product AS p JOIN inventory AS i ON p.product_id = i.product_id WHERE p.product_id ILIKE '%175/65R14%'\n"
    "- SELECT * FROM claim WHERE status ILIKE '%approved%'\n"
    "- SELECT w.location, i.quantity FROM inventory AS i JOIN product AS p ON p.product_id = i.product_id JOIN warehouse AS w ON w.warehouse_id = i.warehouse_id WHERE p.product_id ILIKE '%195/55R15%'\n\n"

    "strictly Product names include terms like 'Speedocruze Pro', 'Agrotuff Cross', etc., while product IDs are sizes like '175/65R14'.\n"
    "Only generate SQL SELECT statements. If the question cannot be answered with the schema, respond with 'NO_SQL'."
)


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": corrected_query}
        #{"role": "user", "content": user_query}
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
        "If the context is empty or says 'No results found.', say so. "
        "Do not invent data or refer to information not present in the context. "
        "Be concise and direct in your answer."
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
    
def clean_sql_output(raw_sql):
    # Remove markdown code fences and leading/trailing spaces
    cleaned = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
    return cleaned

def enhanced_metadata_filter_matching(metadata_filter, row_metadata):
    """
    Enhanced metadata matching with fuzzy logic
    """
    if not metadata_filter or not row_metadata:
        return 0
    
    match_score = 0
    total_filters = len(metadata_filter)
    
    for key, filter_value in metadata_filter.items():
        row_value = row_metadata.get(key, "")
        
        if not filter_value or not row_value:
            continue
            
        # Exact match
        if str(filter_value).lower() == str(row_value).lower():
            match_score += 1
        # Fuzzy match for strings
        elif isinstance(filter_value, str) and isinstance(row_value, str):
            similarity = fuzz.ratio(filter_value.lower(), row_value.lower())
            if similarity >= 70:  # 70% similarity threshold
                match_score += similarity / 100  # Weighted score
        # Partial match
        elif str(filter_value).lower() in str(row_value).lower() or str(row_value).lower() in str(filter_value).lower():
            match_score += 0.7
    
    return match_score / total_filters if total_filters > 0 else 0

def vector_store_similarity_search(query_embedding, top_k=10, metadata_filter=None, similarity_threshold=0.1):
    """
    Enhanced vector similarity search with fuzzy metadata matching
    """
    response = supabase.table("vector_store").select("*").execute()
    rows = response.data
    if not rows:
        return []
    
    # Enhanced metadata filtering with fuzzy matching
    if metadata_filter:
        scored_rows = []
        for row in rows:
            try:
                meta = row.get("metadata")
                if isinstance(meta, str):
                    meta = json.loads(meta)
                
                match_score = enhanced_metadata_filter_matching(metadata_filter, meta)
                if match_score > 0:
                    scored_rows.append((match_score, row))
            except Exception:
                continue
        
        # Sort by metadata match score and take top matches
        scored_rows.sort(reverse=True, key=lambda x: x[0])
        if scored_rows:
            # Take rows with score above threshold (e.g., 30% match)
            rows = [row for score, row in scored_rows if score >= 0.3]
        else:
            rows = []
    
    # Vector similarity with threshold filtering
    similarities = []
    for row in rows:
        try:
            row_embedding = json.loads(row["embedding"])
            sim = cosine_similarity(query_embedding, row_embedding)
            if sim >= similarity_threshold:
                similarities.append((sim, row))
        except Exception:
            continue
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    results = [row for sim, row in similarities[:top_k]]
    
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
        "You are an AI assistant for the tyre manufacturing industry. "
        "Your name is 'shivam'. "
        "You are given two sources of information: SQL database results and retrieved knowledge from a vector store. "
        "Use both sources to answer the user's query as accurately as possible. "
        "If both are relevant, combine the information. If only one is relevant, use that. "
        "If neither is relevant, say so and do not infer anything not present in the context."
        "Act like a helpful assistant - respond naturally without mentioning the sources. "
        "Do not invent data or refer to information not present in the context."
        "Consider if there are speeling mistakes from context provided and in user query , just be careful it should be relevant and proper."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"SQL Context:\n{sql_context}\n\nRAG Context:\n{rag_context}\n\nUser Query: {user_query}"}
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
    Enhanced metadata extraction with fuzzy correction
    """
    corrected_query = fuzzy_correct_entities(user_query)
    
    system_prompt = (
        "You are an assistant that extracts structured metadata from user queries about tyres, warehouses, dealers, claims, sales, and inventory. "
        "Given a user query, return a JSON object with as many of these fields as possible if present: "
        "claim_id, dealer_id, dealer_name, status, product_id, product_name, category, warehouse_id, location, zone, quantity, sales_id, date, cost, product_price, warehouse_location. "
        "Be flexible with variations and partial matches in names and locations. "
        "If a field is not present, omit it. Only return a valid JSON object, no explanation or extra text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": corrected_query}
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
    Enhanced main function with comprehensive fuzzy matching
    """
    print("Enhanced RAG System with Fuzzy Matching for Spelling Mistakes")
    print("Initializing entity cache...")
    
    # Initialize entity cache for fuzzy matching
    get_database_entities()
    
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_query = input("query : ")
        if user_query.lower() == "exit":
            break
        
        try:
            # Step 1: Enhanced SQL Generation with fuzzy correction
            sql = get_llm_sql(user_query)
            sql_context = "No results found."
            sql = clean_sql_output(sql)
            print("DEBUG: Cleaned SQL:", sql)

            print("DEBUG: Corrected user query:", fuzzy_correct_entities(user_query))
            print("DEBUG: LLM raw SQL response:", sql)

            
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
                print("SQL Generation: Only SELECT statements are allowed.")

            # Step 2: Enhanced RAG pipeline with fuzzy matching
            print("\n" + "="*60)
            print("RAG Pipeline with Fuzzy Matching")
            print("="*60)
            
            # Enhanced query rewriting with fuzzy correction
            rewritten_query = rewrite_query_for_rag(user_query)
            preprocessed_query = preprocess_query(rewritten_query)
            query_embedding = get_embedding(preprocessed_query)
            
            # Extract metadata with fuzzy correction
            metadata_filter = extract_metadata_with_llm(user_query)
            print("DEBUG: metadata_filter =", metadata_filter)
            
            # Enhanced vector search with fuzzy metadata matching
            vector_rows = vector_store_similarity_search(
                query_embedding, 
                top_k=10, 
                metadata_filter=metadata_filter,
                similarity_threshold=0.08  # Lower threshold for better recall
            )
            
            rag_context = vector_rows_to_context(vector_rows) if vector_rows else "No relevant vector context found."
            
            if vector_rows:
                print("RAG Vector Search Result:")
                print(rag_context)
            else:
                print("RAG Vector Search: No relevant results found.")

            # Step 3: Generate final response
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