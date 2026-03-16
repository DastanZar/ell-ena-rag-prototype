#!/usr/bin/env python3
"""
Lightweight RAG Context-Filtering Prototype

This script compares a meeting transcript against existing tickets using TF-IDF
and cosine similarity, then uses an LLM to extract new action items.
"""

import json
import sys
from pathlib import Path

# Third-party imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import OpenAI (preferred), fallback to google.generativeai
try:
    import openai
    LLM_PROVIDER = "openai"
except ImportError:
    try:
        import google.generativeai as genai
        LLM_PROVIDER = "google"
    except ImportError:
        LLM_PROVIDER = None

# Configuration - UPDATE THIS WITH YOUR API KEY
API_KEY = "Use any free key from Google AI studio"  # Replace with your actual API key
MODEL_NAME = "gpt-4" if LLM_PROVIDER == "openai" else "gemini-pro"


def load_transcript(file_path: str) -> str:
    """Load transcript text from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Transcript file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading transcript: {e}")
        sys.exit(1)


def load_tickets(file_path: str) -> list[dict]:
    """Load tickets from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Tickets file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing tickets JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading tickets: {e}")
        sys.exit(1)


def find_top_relevant_tickets(transcript: str, tickets: list[dict], top_n: int = 2) -> list[dict]:
    """
    Use TF-IDF and cosine similarity to find the most relevant tickets
    based on the transcript text.
    """
    if not tickets:
        print("Warning: No tickets to compare against")
        return []
    
    # Extract ticket titles
    ticket_titles = [ticket.get("title", "") for ticket in tickets]
    
    # Handle empty titles
    ticket_titles = [t if t else f"Ticket {i+1}" for i, t in enumerate(ticket_titles)]
    
    # Combine transcript with ticket titles for TF-IDF vectorization
    documents = [transcript] + ticket_titles
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity between transcript (first doc) and all tickets
        transcript_vector = tfidf_matrix[0:1]
        ticket_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(transcript_vector, ticket_vectors).flatten()
        
        # Get indices of top N most similar tickets
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return top N tickets with their similarity scores
        top_tickets = []
        for idx in top_indices:
            ticket = tickets[idx].copy()
            ticket['_similarity_score'] = float(similarities[idx])
            top_tickets.append(ticket)
        
        return top_tickets
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        # Fallback: return first N tickets
        return tickets[:top_n]


def create_prompt(transcript: str, relevant_tickets: list[dict]) -> str:
    """Create the prompt for the LLM to extract new action items."""
    
    # Format existing tickets for the prompt
    existing_tickets_text = json.dumps(relevant_tickets, indent=2)
    
    prompt = f"""You are an AI assistant that extracts action items from meeting transcripts.

## Existing Relevant Tickets (for context - DO NOT include these as new items):
{existing_tickets_text}

## Meeting Transcript:
{transcript}

## Your Task:
Analyze the transcript above and extract ONLY the NEW action items that are NOT already covered by the existing tickets.

For each new action item, extract the following information:
- Assignee: Who is responsible for this task?
- Deadline: When is this task due? (format as specified in transcript, or "No deadline specified")
- Task Title: A brief description of the task
- Priority: High, Medium, or Low (based on urgency mentioned in transcript)

## Important Rules:
1. ONLY extract tasks that are NOT already in the existing tickets
2. If a task in the transcript is already covered by an existing ticket, do NOT include it
3. Extract ALL new tasks mentioned in the transcript
4. Output ONLY a JSON array of objects with the exact keys: Assignee, Deadline, Task Title, Priority
5. Do NOT include any additional text or explanations
6. Use null for any missing information

## Output Format:
```json
[
  {{
    "Assignee": "Name",
    "Deadline": "Date or description",
    "Task Title": "Task description",
    "Priority": "High/Medium/Low"
  }}
]
```

Now extract the new action items:"""

    return prompt


def call_llm(prompt: str) -> str:
    """Call the LLM API to extract action items."""
    
    if LLM_PROVIDER is None:
        return generate_mock_response()
    
    try:
        if LLM_PROVIDER == "openai":
            openai.api_key = API_KEY
            
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts action items from meeting transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif LLM_PROVIDER == "google":
            genai.configure(api_key=API_KEY)
            
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        print("Falling back to mock response...")
        return generate_mock_response()


def generate_mock_response() -> str:
    """
    Generate a mock response when API is not available.
    This simulates what the LLM would return.
    """
    # Based on the transcript, these would be the new items:
    mock_response = [
        {
            "Assignee": "Mike",
            "Deadline": "Friday",
            "Task Title": "Fix frontend 401 error handling in login screen",
            "Priority": "High"
        },
        {
            "Assignee": "Sarah",
            "Deadline": "Wednesday",
            "Task Title": "Review backend token lifecycle for OAuth expiration",
            "Priority": "High"
        }
    ]
    return json.dumps(mock_response, indent=2)


def parse_llm_response(response: str) -> list[dict]:
    """Parse the LLM's JSON response."""
    try:
        # Try to extract JSON from response (in case of markdown formatting)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {response}")
        return []


def print_results(top_tickets: list[dict], action_items: list[dict]) -> None:
    """Print the selected tickets and final action items."""
    print("\n" + "="*60)
    print("TOP 2 SELECTED CONTEXT TICKETS")
    print("="*60)
    
    for i, ticket in enumerate(top_tickets, 1):
        print(f"\n{i}. [{ticket.get('id', 'N/A')}] {ticket.get('title', 'No title')}")
        print(f"   Status: {ticket.get('status', 'N/A')}")
        print(f"   Similarity Score: {ticket.get('_similarity_score', 0):.4f}")
    
    print("\n" + "="*60)
    print("NEW ACTION ITEMS EXTRACTED BY LLM")
    print("="*60)
    
    if action_items:
        print(json.dumps(action_items, indent=2))
    else:
        print("No new action items found.")
    
    print("="*60 + "\n")


def main():
    """Main function to run the RAG context-filtering prototype."""
    
    # File paths
    transcript_path = "transcript.txt"
    tickets_path = "tickets.json"
    
    print("="*60)
    print("RAG Context-Filtering Prototype")
    print("="*60)
    
    # Step 1: Load data
    print("\n[1/5] Loading transcript and tickets...")
    transcript = load_transcript(transcript_path)
    tickets = load_tickets(tickets_path)
    print(f"   Loaded transcript ({len(transcript)} chars)")
    print(f"   Loaded {len(tickets)} tickets")
    
    # Step 2: Find top relevant tickets using TF-IDF + cosine similarity
    print("\n[2/5] Finding top 2 relevant tickets using TF-IDF...")
    top_tickets = find_top_relevant_tickets(transcript, tickets, top_n=2)
    print(f"   Selected {len(top_tickets)} relevant tickets")
    
    # Step 3: Create prompt for LLM
    print("\n[3/5] Creating prompt for LLM...")
    prompt = create_prompt(transcript, top_tickets)
    print(f"   Prompt created ({len(prompt)} chars)")
    
    # Step 4: Call LLM
    print("\n[4/5] Calling LLM API...")
    if LLM_PROVIDER == "openai":
        print(f"   Using OpenAI ({MODEL_NAME})")
    elif LLM_PROVIDER == "google":
        print(f"   Using Google Generative AI ({MODEL_NAME})")
    else:
        print("   No LLM provider available - using mock response")
    
    llm_response = call_llm(prompt)
    print(f"   Received response ({len(llm_response)} chars)")
    
    # Step 5: Parse and display results
    print("\n[5/5] Processing and displaying results...")
    action_items = parse_llm_response(llm_response)
    print_results(top_tickets, action_items)
    
    print("Done!")


if __name__ == "__main__":
    main()
