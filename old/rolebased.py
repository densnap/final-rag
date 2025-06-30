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

# Global user session
current_user = None

# Cache for fuzzy matching
DEALER_CACHE = {}
PRODUCT_CACHE = {}
WAREHOUSE_CACHE = {}

class UserSession:
    def __init__(self, user_id, username, role, dealer_id=None, dealer_name=None):
        self.user_id = user_id
        self.username = username
        self.role = role.lower()
        self.dealer_id = dealer_id
        self.dealer_name = dealer_name
        self.is_authenticated = True
    
    def is_dealer(self):
        return self.role == 'dealer'
    
    def is_admin(self):
        return self.role in ['admin', 'superuser', 'manager']
    
    def can_access_all_data(self):
        return self.is_admin()
    
    def get_dealer_filter(self):
        """Returns dealer_id for filtering if user is a dealer"""
        return self.dealer_id if self.is_dealer() else None

def authenticate_user(username, password):
    """
    Authenticate user and return UserSession object
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("dbname"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port")
        )
        cur = conn.cursor()
        
        # Get user details with dealer info
        query = """
        SELECT u.user_id, u.username, u.role, u.dealer_id, d.name as dealer_name
        FROM users u
        LEFT JOIN dealer d ON u.dealer_id = d.dealer_id
        WHERE u.username = %s AND u.password = %s
        """
        cur.execute(query, (username, password))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            user_id, username, role, dealer_id, dealer_name = result
            return UserSession(user_id, username, role, dealer_id, dealer_name)
        else:
            return None
            
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def login():
    """
    Handle user login
    """
    global current_user
    
    print("=== LOGIN REQUIRED ===")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    user_session = authenticate_user(username, password)
    
    if user_session:
        current_user = user_session
        print(f"Welcome {user_session.username}!")
        print(f"Role: {user_session.role}")
        if user_session.dealer_name:
            print(f"Dealer: {user_session.dealer_name}")
        print("-" * 50)
        return True
    else:
        print("Invalid credentials. Please try again.")
        return False

def logout():
    """
    Handle user logout
    """
    global current_user
    if current_user:
        print(f"Goodbye {current_user.username}!")
        current_user = None

def check_authentication():
    """
    Check if user is authenticated
    """
    return current_user is not None and current_user.is_authenticated

def normalize_text(text):
    """Normalize text for better matching"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
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
    
    try:
        result = process.extractOne(query_normalized, candidates_normalized, scorer=fuzz.ratio)
        if result and result[1] >= threshold:
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
            if len(word) > 3:
                text_words = text.lower().split()
                for text_word in text_words:
                    similarity = fuzz.ratio(word, text_word)
                    if similarity >= 75 and word != text_word:
                        pattern = re.compile(re.escape(text_word), re.IGNORECASE)
                        corrected_text = pattern.sub(word, corrected_text)
                        corrections_made.append(f"{text_word} -> {word}")
    
    # Check for product IDs
    for cached_product in PRODUCT_CACHE.values():
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

def add_role_based_filters(base_query, user_session):
    """
    Add role-based access control filters to SQL queries
    """
    if not user_session or user_session.can_access_all_data():
        return base_query
    
    if not user_session.is_dealer():
        return base_query
    
    # For dealers, add filters based on query type
    query_lower = base_query.lower()
    dealer_id = user_session.dealer_id
    
    # If query involves sales table
    if 'from sales' in query_lower or 'join sales' in query_lower:
        if 'where' in query_lower:
            # Add dealer filter to existing WHERE clause
            base_query = base_query.replace(' WHERE ', f' WHERE s.dealer_id = {dealer_id} AND ')
            base_query = base_query.replace(' where ', f' WHERE s.dealer_id = {dealer_id} AND ')
        else:
            # Add WHERE clause with dealer filter
            base_query = base_query.rstrip(';') + f' WHERE s.dealer_id = {dealer_id};'
    
    # If query involves claims table
    elif 'from claim' in query_lower or 'join claim' in query_lower:
        if 'where' in query_lower:
            base_query = base_query.replace(' WHERE ', f' WHERE c.dealer_id = {dealer_id} AND ')
            base_query = base_query.replace(' where ', f' WHERE c.dealer_id = {dealer_id} AND ')
        else:
            base_query = base_query.rstrip(';') + f' WHERE c.dealer_id = {dealer_id};'
    
    # Inventory queries are allowed for all dealers (they can see all stock)
    
    return base_query

def rewrite_query_for_rag(user_query):
    """
    Enhanced query rewriting with fuzzy correction
    """
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
    """Enhanced SQL generation with role-based access control"""
    corrected_query = fuzzy_correct_entities(user_query)
    
    # Enhanced system prompt with role-based considerations
    role_info = ""
    if current_user and current_user.is_dealer():
        role_info = f"""
        
IMPORTANT ROLE-BASED ACCESS CONTROL:
- The current user is a DEALER with dealer_id = {current_user.dealer_id}
- Dealers can see:
  * All inventory/stock data (no restrictions)
  * Only their own sales data (filter by dealer_id = {current_user.dealer_id})
  * Only their own claims (filter by dealer_id = {current_user.dealer_id})
- For sales queries: ALWAYS add "AND s.dealer_id = {current_user.dealer_id}" to WHERE clause
- For claims queries: ALWAYS add "AND c.dealer_id = {current_user.dealer_id}" to WHERE clause
- For inventory queries: No dealer restrictions needed
"""
    
    system_prompt = (
        "You are an AI assistant for a tyre manufacturing company. "
        "You help generate efficient SQL SELECT queries from user questions based on the following PostgreSQL schema.\n\n"
        "Tables and their descriptions:\n"
        "1. users(user_id, username, password, email, role, dealer_id): Stores user login and profile information, including their role and associated dealer.\n"
        "2. dealer(dealer_id, name): Stores information about tyre dealers.\n"
        "3. claim(claim_id, dealer_id, status): Stores warranty or service claims made by dealers.\n"
        "4. product(product_id, product_name, category, price, section_width, aspect_ratio, construction_type, rim_diameter_inch): Stores product (tyre) details, including size and pricing.\n"
        "5. warehouse(warehouse_id, location, zone): Stores warehouse locations and zones.\n"
        "6. sales(sales_id, dealer_id, product_id, warehouse_id, quantity, cost, date): Stores sales transactions, including which dealer sold which product from which warehouse, quantity, cost, and date.\n"
        "7. inventory(product_id, warehouse_id, quantity): Stores current stock levels of each product in each warehouse.\n\n"
        "Key relationships:\n"
        "- users joins dealer on dealer_id\n"
        "- claim joins dealer on dealer_id\n"
        "- sales joins dealer, product, and warehouse via dealer_id, product_id, and warehouse_id\n"
        "- inventory joins product and warehouse via product_id and warehouse_id\n\n"
        + role_info +
        "do not create any sql queries if informaton of other than dealer_id = {current_user.dealer_id} is asked.\n "
        "\nRules for Generating SQL:\n"
        "1. Use ONLY the columns and tables exactly as defined above.\n"
        "2. For string comparisons (names, locations, products), use ILIKE with % wildcards for partial matching to handle typos: WHERE name ILIKE '%searchterm%'\n"
        "3. Use LOWER() for case-insensitive comparisons\n"
        "4. When searching for names or locations, try multiple variations\n"
        "5. Use OR conditions to check multiple possible spellings\n"
        "6. Always SELECT and return all columns necessary to answer the user's question\n"
        "7. Use meaningful aliases (s for sales, c for claim, d for dealer, p for product, w for warehouse, i for inventory)\n"
        "8. STRICTLY ENFORCE role-based access control as specified above\n\n"
        "Only generate SQL SELECT statements. If the question cannot be answered with the schema, respond with 'NO_SQL'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": corrected_query}
    ]
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 250,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post(chat_endpoint, headers=chat_headers, json=payload)
    if response.status_code == 200:
        sql = response.json()["choices"][0]["message"]["content"].strip()
        
        # Additional safety check - apply role-based filters
        if current_user:
            sql = add_role_based_filters(sql, current_user)
        
        return sql
    else:
        raise Exception(f"Chat API error (SQL): {response.status_code}: {response.text}")
    
def clean_sql_output(raw_sql):
    # Remove markdown code fences and leading/trailing spaces
    cleaned = re.sub(r"```(?:sql)?", "", raw_sql, flags=re.IGNORECASE).strip()
    return cleaned

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

def enhanced_metadata_filter_matching(metadata_filter, row_metadata):
    """
    Enhanced metadata matching with fuzzy logic and role-based filtering
    """
    if not metadata_filter or not row_metadata:
        return 0
    
    # Add role-based filtering to metadata
    if current_user and current_user.is_dealer():
        # Check if metadata contains dealer-specific information
        if 'dealer_id' in row_metadata:
            if str(row_metadata['dealer_id']) != str(current_user.dealer_id):
                # If this is sales or claims data from another dealer, exclude it
                if any(key in row_metadata for key in ['sales_id', 'claim_id']):
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
            if similarity >= 70:
                match_score += similarity / 100
        # Partial match
        elif str(filter_value).lower() in str(row_value).lower() or str(row_value).lower() in str(filter_value).lower():
            match_score += 0.7
    
    return match_score / total_filters if total_filters > 0 else 0

def vector_store_similarity_search(query_embedding, top_k=10, metadata_filter=None, similarity_threshold=0.1):
    """
    Enhanced vector similarity search with role-based access control
    """
    response = supabase.table("vector_store").select("*").execute()
    rows = response.data
    if not rows:
        return []
    
    # Enhanced metadata filtering with role-based access control
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
        
        scored_rows.sort(reverse=True, key=lambda x: x[0])
        if scored_rows:
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
    user_context = ""
    if current_user:
        user_context = (
            f"\nUser Context:\n"
            f"- Username: {current_user.username}\n"
            f"- Role: {current_user.role}\n"
            f"- dealer_id: {current_user.dealer_id}\n"
        )
        if current_user.is_dealer():
            user_context += (
                f"- Dealer Name: {current_user.dealer_name}\n"
                f"- Dealer ID: {current_user.dealer_id}\n"
                "IMPORTANT RESTRICTION:\n"
                "This user is a DEALER and must only see their own data.\n"
                "Do NOT include or mention sales or claims from ANY other dealer.\n"
                "If the SQL or RAG context includes such data, ignore or exclude it.\n"
            )

    system_prompt = (
        "You are an AI assistant for the tyre manufacturing industry. "
        "Your name is 'shivam'.\n"
        "You are given two sources of information: SQL database results and retrieved knowledge from a vector store (RAG).\n"
        "Always prioritize the SQL data, and ensure your response includes all relevant SQL results.\n"
        "If both SQL and RAG are relevant, combine them. If only one is relevant, use that.\n"
        "NEVER fabricate data. NEVER include information not present in the context.\n"
        "STRICTLY ENFORCE role-based visibility:\n"
        "- If the logged-in user is a dealer, NEVER mention sales or claims made by other dealers.\n"
        "- Only show data tied to the dealer ID mentioned below.\n"
        "Give direct answer without any explanations or additional information.\n"
        "If the user asks for information not present in the context, respond with 'Sorry, I can't assist with that.'\n"
        "If no answer can be generated, respond with 'Sorry, I can't assist with that.'\n"
        + user_context
        
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
    Enhanced metadata extraction with role-based context
    """
    corrected_query = fuzzy_correct_entities(user_query)
    
    # Add current user context to metadata extraction
    user_context = ""
    if current_user and current_user.is_dealer():
        user_context = f" The current user is dealer {current_user.dealer_name} with dealer_id {current_user.dealer_id}."
    
    system_prompt = (
        "You are an assistant that extracts structured metadata from user queries about tyres, warehouses, dealers, claims, sales, and inventory. "
        "Given a user query, return a JSON object with as many of these fields as possible if present: "
        " user_id, username, email, role, dealer_id, dealer_name,claim_id, status, claim_date, product_id(number ,symbol , alphabets eg. 100/35R24 50P), amount, approved_amount, resolved_date, reason,sales_id, date, product_name(only text eg. SpeedoCruze Pro), warehouse_id(number),zone,quantity(number), cost, product_price, category, location"
        "Be flexible with variations and partial matches in names and locations. "
        + user_context +
        " If a field is not present, omit it. Only return a valid JSON object, no explanation or extra text."
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
    Enhanced main function with role-based access control
    """
    print("🔐 Enhanced RAG System with Role-Based Access Control")
    print("Features: Fuzzy Matching + Dealer Access Control")
    print("=" * 60)
    
    # Authentication loop
    while not check_authentication():
        if not login():
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                print("Goodbye!")
                return
    
    # Initialize entity cache
    print("Initializing entity cache...")
    get_database_entities()
    
    print(f"\nWelcome to the RAG system, {current_user.username}!")
    if current_user.is_dealer():
        print("🏪 Dealer Access: You can see all inventory, but only your sales and claims")
    elif current_user.is_admin():
        print("🔑 Admin Access: You have full access to all data")
    
    print("\nCommands:")
    print("- Type your query to search")
    print("- Type 'logout' to switch users")
    print("- Type 'exit' to quit")
    print("-" * 60)
    
    while True:
        user_query = input(f"{current_user.username} > ").strip()
        
        if user_query.lower() == "exit":
            break
        elif user_query.lower() == "logout":
            logout()
            main()  # Restart the main loop
            return
        elif not user_query:
            continue
        
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