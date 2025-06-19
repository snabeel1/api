import os
import re
import uuid
import unicodedata
import json
from datetime import datetime

from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct 
from sentence_transformers import SentenceTransformer


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) 

FAQ_COLLECTION_NAME = "amlak_faqs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL_ID = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBED_MODEL_ID)

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)


CATEGORY_SLUG_MAP = {
    "Property Ownership": "ownership",
    "Community Zones & Freehold Areas": "communities",
    "Off-Plan Property Questions": "off-plan",
    "Developer-Specific FAQs": "developers",
    "Legal FAQs": "legal",
    "Pricing & ROI": "prices",
    "Rental & Property Management": "rental",
    "Property Buying Process": "buying-process",
    "Visa, Tax & Inheritance": "visa-tax-inheritance",
    "APIL GPT Specific FAQs": "apil-gpt"
}

def generate_short_slug(text, max_words=5):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    words = [word for word in text.split() if word not in ["is", "the", "a", "an", "of", "in", "for", "to", "how", "what", "can", "are", "by", "what's", "do", "does", "or", "and", "vs", "which", "will", "are", "its"]]
    
    short_slug = "-".join(words[:max_words])
    short_slug = re.sub(r'[-\s]+', '-', short_slug).strip('-')
    return short_slug

def get_embedding(text: str) -> list[float]:
    return embedding_model.encode(text).tolist()

def search_faqs(query: str, top_k: int = 2, category_filter: str = None):
    try:
        query_vector = get_embedding(query)
        qdrant_filter = None
        if category_filter:
            if category_filter in CATEGORY_SLUG_MAP.values():
                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category_slug", 
                            match=models.MatchValue(value=category_filter)
                        )
                    ]
                )
            else:
                print(f"WARNING: Invalid category_filter '{category_filter}' provided. Skipping filter.")
        
        search_results = qdrant_client.search(
            collection_name=FAQ_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_args=models.QueryArgs(
                filter=qdrant_filter # Apply the filter here (will be None if no valid filter)
            )
        )
        
        faqs = []
        for hit in search_results:
            faqs.append({
                "question": hit.payload.get("question"),
                "answer": hit.payload.get("answer"),
                "category": hit.payload.get("category"),
                "full_url_slug": hit.payload.get("full_url_slug"),
                "score": hit.score # For debugging/understanding relevance
            })
        
        return faqs
    except Exception as e:
        print(f"Error searching FAQs: {e}")
        return []

faq_search_tool = {
    "type": "function",
    "function": {
        "name": "search_faqs",
        "description": f"""Searches for answers to general real estate questions from the FAQ knowledge base.
        Use this tool when the user asks a 'how-to', 'what-is', 'can-I', 'who-is', or 'is-it' type of question about real estate in Dubai/UAE.
        This is the primary function of the assistant.
        
        You can optionally filter the search to specific categories. Only use a category filter if the user explicitly mentions a category or a highly specific topic that directly maps to one of these: {list(CATEGORY_SLUG_MAP.keys())}.
        The corresponding slug values for filtering are: {list(CATEGORY_SLUG_MAP.values())}.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question or topic of interest to search within the FAQs."},
                "category_filter": {"type": "string", "enum": list(CATEGORY_SLUG_MAP.values()), "description": "Optional: Filter FAQs by a specific category slug (e.g., 'visa-tax-inheritance', 'rental'). Only use if the user explicitly specifies or implies a strong category preference."}
            },
            "required": ["query"],
        },
    },
}

available_tools = [faq_search_tool]

def generate_response(user_input: str, chat_history: list):
    messages = [
        {"role": "system", "content": f"""You are Amlak, a highly knowledgeable and helpful AI assistant for Dubai and UAE real estate FAQs.
        Your primary purpose is to answer general questions about Dubai/UAE real estate, including topics like property ownership, off-plan projects, legal aspects, taxes, visas, rental processes, and developer information.

        When answering questions, use information retrieved from the FAQ knowledge base. Be concise but informative. Always provide the answer found. If a relevant FAQ is found and has a corresponding URL slug, incorporate a direct link into your response for the user to find more details (e.g., "You can find more details here: [link]").

        If you cannot find a direct answer in your knowledge base, state that clearly and politely, and offer to search for other related topics or rephrase the question.

        Current date and time: {datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")}
        Current location: Dubai, Dubai, United Arab Emirates
        """}
    ] + chat_history + [{"role": "user", "content": user_input}]

    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o", # Using a powerful model for best tool calling
            messages=messages,
            tools=available_tools,
            tool_choice="auto", 
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_faqs":
                    query_for_faq = function_args.get("query")
                    category_filter_arg = function_args.get("category_filter") 
                    
                    print(f"DEBUG: Calling search_faqs with query: '{query_for_faq}', category_filter: '{category_filter_arg}'")
                    
                    faqs_found = search_faqs(query_for_faq, category_filter=category_filter_arg) 
                    
                    if faqs_found:
                        top_faq = faqs_found[0]
                        answer = top_faq['answer']
                        full_url_slug = top_faq['full_url_slug']
                        
                        response_text = f"Regarding your question about '{top_faq['question']}', here's what I found:\n\n{answer}"
                        if full_url_slug:
                            response_text += f"\n\nFor more details, you can visit our website: [website]{full_url_slug}[/website]" 
                        
                        return response_text
                    else:
                        return "I couldn't find a direct answer to that specific question in my FAQ knowledge base. Can you please rephrase or ask about a different topic?"
                else:
                    return "I'm sorry, I encountered an unexpected tool call."
        else:
            return response_message.content

    except Exception as e:
        print(f"Error in generating response: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again later."   
if __name__ == "__main__":
    chat_history = []
    print("Amlak FAQ Assistant (Type 'exit' to quit)")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        
        # Limit chat history to recent interactions to keep context concise
        # (e.g., last 4 turns: 2 user, 2 assistant messages)
        # This helps manage token usage for the LLM
        if len(chat_history) > 4: 
            chat_history = chat_history[-4:]

        response = generate_response(user_query, chat_history)
        print(f"Amlak AI: {response}")
        
        # Append current interaction to history for the next turn
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response})
