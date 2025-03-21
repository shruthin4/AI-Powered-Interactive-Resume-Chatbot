import os
import re
import logging
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

from chroma_db import (
    query_chromadb,
    list_chromadb_documents,
    detect_folder_category,
    combine_docs_text
)


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Flask_Backend")

SESSIONS = {}

@app.route("/ShruthinReddy")
def website():
    return render_template("intro.html")

@app.route("/list_docs", methods=["GET"])
def list_docs():
    """Lists all stored documents, including notes, links, images, etc."""
    stored_docs = list_chromadb_documents()
    return jsonify({"stored_documents": stored_docs})


def get_session_id(request):
    """
    Retrieve session_id from request (query param or JSON).
    Fallback to "default_session" if none provided.
    """
    sid = request.args.get("session_id")
    if not sid and request.is_json:
        data = request.get_json()
        sid = data.get("session_id", None)
    return sid if sid else "default_session"


def is_greeting(message):
    """Return True if the user’s message is just a greeting."""
    message = message.lower().strip()
    greetings = ["hi", "hello", "hey", "greetings", "howdy", "hola"]
    return message in greetings


def is_notes_request(message):
    """Checks if the user is asking for notes or cheat sheets."""
    message_lower = message.lower().strip()
    notes_keywords = [
        "notes", "cheat sheets", "what notes do you have?", "show me the notes",
        "list notes", "available notes", "show notes", "list all notes",
        "what notes are available", "do you have any notes"
    ]
    return any(phrase == message_lower for phrase in notes_keywords)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    session_id = get_session_id(request)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}

    logger.info(f"[Session {session_id}] User: {user_message}")

    if is_greeting(user_message):
        bot_reply = "<p>Hello! How can I help you today?</p>"
        SESSIONS[session_id]["history"].append({"user": user_message, "bot": bot_reply})
        return jsonify({"reply": bot_reply})

    if is_notes_request(user_message):
        notes_resp = retrieve_notes_response()
        resp_data = notes_resp.get_json()  
        SESSIONS[session_id]["history"].append({
            "user": user_message,
            "bot": resp_data["reply"]
        })
        return notes_resp

    folder_cat = detect_folder_category(user_message)

    docs = query_chromadb(user_message, folder_category=folder_cat)
    logger.info(f"[Session {session_id}] Retrieved {len(docs)} docs.")

    if not docs:
        bot_reply = "<p>No relevant documents found about Shruthin.</p>"
        SESSIONS[session_id]["history"].append({"user": user_message, "bot": bot_reply})
        return jsonify({"reply": bot_reply})

    merged_text = combine_docs_text(docs)

    conversation_history = build_history_prompt(SESSIONS[session_id]["history"])
    final_html = generate_llm_response(user_message, merged_text, conversation_history)

    final_html = cleanup_html(final_html)

    SESSIONS[session_id]["history"].append({"user": user_message, "bot": final_html})
    return jsonify({"reply": final_html})


def build_history_prompt(history):
    """Keep short conversation context from last 2 turns."""
    lines = []
    for turn in history[-2:]:
        lines.append(f"User said: {turn['user']}\nBot said: {turn['bot']}\n")
    return "\n".join(lines)


def retrieve_notes_response():

    logger.info("📜 Notes request detected.")
    stored = list_chromadb_documents()
    notes_files = [d["source"] for d in stored if d["category"].lower() == "notes"]
    if not notes_files:
        return jsonify({"reply": "<p>No notes or cheat sheets available from Shruthin.</p>"})

    response_text = "<h2>Available Notes & Cheat Sheets</h2><ul>"
    for note in notes_files:
        display_name = os.path.splitext(note)[0].replace("_", " ").replace("-", " ")
        response_text += f"<li><strong>{display_name}</strong></li>"
    response_text += "</ul><p>Which notes would you like to see in detail?</p>"

    return jsonify({"reply": response_text})


def generate_llm_response(user_message, structured_content, conversation_history=""):

    prompt = f"""
Here is our short conversation so far:
{conversation_history}

You are an AI assistant created by Shruthin Reddy. Always speak in the first person as if you are Shruthin.

PERSONALITY & TONE INSTRUCTIONS:
1. Use a natural, conversational tone without forced friendliness
2. Avoid repetitive greeting patterns like "Hey there! So you want..."
3. Start responses directly with the information requested 
4. Only include casual closings like "Let me know if you need anything else" occasionally, not in every response
5. Use contractions (I'm, I've, I'd) naturally
6. Never say "So you want to know about..." - just provide the information directly
7. Vary your response patterns - don't follow the same sentence structure each time
8. Use closers occasionally but vary them - don't use the exact same phrase every time

INTRODUCTION VARIATIONS: They can be varied below are just examples
- For skills: "My skill set includes:" or "I'm proficient in several areas:"
- For education: "I've studied at:" or "My academic background includes:"
- For projects: "I've worked on several projects including:" or "Some of my notable projects are:"

RESPONSE STRUCTURE GUIDELINES:
1. Start with a brief, natural introduction (1 short sentence) - vary this introduction across responses
2. Avoid repetitive phrases like "Here's a summary of my skills" or "Here's my education background"
3. For category-specific information (skills, education, etc.), use clear headers in <h3> tags
4. For skills, always organize into categories with proper headings (<h3>)
5. Use consistent bullet formatting for ALL list items
6. NEVER end responses with "Let me know if you need anything else" or similar phrases. End responses naturally without forced friendliness 

CONTENT RULES:
1. If the user asks about "Skills" → ONLY discuss skills
2. If the user asks about "Certifications" → ONLY discuss certifications
3. If user asks for verification links → PROVIDE verification links and IDs as PLAIN TEXT for available Certificates
4. If user asks for certification IDs → PROVIDE certificate IDs in plain text format "Certificate ID: 12345ABC"
5. If asked for notes → Show full content with proper formatting
6. Check that Microsoft Azure Fundamentals (AZ-900) is included in certification lists
7. Don't introduce yourself in each response
8. When showing SQL vs PySpark vs Pandas cheat sheet, include ALL operations and examples
9. NEVER add unrelated information to the end of a response
10.When discussing skills, extract ALL relevant skills  through keywords or semantic meanings from certifications, projects, and work experience
11.NEVER show information that wasn't asked for (don't list certifications when asked for skills)

FORMATTING GUIDELINES:
1. Use bullet points (*) for all lists for consistency
2. For code examples, keep lines under 40 characters and use proper indentation
3. Always wrap code in <pre><code> tags
4. Break long lines of code or queires at commas, dots, or operators
5. Keep responses compact and focused

CERTIFICATION DISPLAY INSTRUCTIONS:
1. When listing certifications, ALWAYS include ALL  certifications retrieved from the database
2. Double-check that exactly ertifications are included in the response, no more or less
3. For certifications with IDs but no links, include "Certificate ID: 12345ABC" (plain text only)
4. ALWAYS include Microsoft Azure Fundamentals certification in the list when showing certifications

CRITICAL HTML FORMATTING INSTRUCTIONS: The below rules are not applied for links they are only for text 
1. Always format your response using proper HTML tags. Output MUST be valid HTML.
2. For emphasis, use <strong>text</strong> tags, not asterisks or other markdown.
3. For lists, always use <ul> and <li> tags:
   <ul>
     <li><strong>Item 1:</strong> Description</li>
     <li><strong>Item 2:</strong> Description</li>
   </ul>
4. For headings, use <h2> and <h3> tags, not asterisks or bold text.
5. For paragraphs, use <p>text</p> tags.

LINK FORMATTING INSTRUCTIONS:
1. NEVER format links with target="_blank" or class="chat-link" attributes
2. For links, use ONLY plain text URLs without ANY HTML tags: https://example.com
3. DO NOT repeat the same URL multiple times in a row
4. NEVER use <a> tags for links, only plain text URLs
5. For verification links, format as "Verification: https://example.com" (plain text only)





The user's query: "{user_message}"

Here are the relevant doc excerpts:
{structured_content}

Format response with proper HTML (headers, paragraphs, code blocks). Remember to be casual and conversational in tone.
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No response."
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "<p>Sorry, I'm unable to process your request right now. Please try again later.</p>"



def cleanup_html(text):
    """Enhanced cleaning for HTML responses to fix link issues."""
    if not text:
        return "<p>No content returned.</p>"
   
    text = re.sub(r'```(\w+)?', '', text)
  
    text = text.replace('`', '')
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:

        if ('target="_blank"' in line or 'class="chat-link"' in line) and 'https://' in line:
            urls = re.findall(r'(https?://[^\s"<>]+)', line)
            
            if urls and len(urls) > 0:
                if '<li>' in line or '* ' in line:
                    text_part = re.sub(r'https?://[^\s"<>]+.*$', '', line)

                    text_part = re.sub(r'"\s+target="_blank"\s+class="chat-link">', '', text_part)
                    url = urls[0]

                    line = f"{text_part} {url}"
                else:
                    line = urls[0]
        
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    

    text = re.sub(
        r'(https?://[^\s"<>]+)"\s+target="_blank"\s+class="chat-link">(https?://[^\s"<>]+)>(https?://[^\s"<>]+)', 
        r'\1', 
        text
    )
    
    text = re.sub(
        r'(https?://[^\s"<>]+)"\s+target="_blank"\s+class="chat-link">(https?://[^\s"<>]+)', 
        r'\1', 
        text
    )
 
    text = re.sub(
        r'(https?://[^\s"<>]+)"\s+target="_blank"\s+class="chat-link">', 
        r'\1', 
        text
    )
    
    text = re.sub(r'<a\s+href="(https?://[^\s"<>]+)">(https?://[^\s"<>]+)</a>', r'\1', text)
 
    text = re.sub(r'\s+target="_blank"', '', text)
    text = re.sub(r'\s+class="chat-link"', '', text)
    text = re.sub(r'href="([^"]+)>"', r'href="\1"', text)
    
    return text


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)














