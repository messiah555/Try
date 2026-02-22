AI-SDR: Autonomous Sales Representative with Hybrid Memory Management
Project Overview
The AI-SDR (Sales Development Representative) is a sophisticated, autonomous agent designed to revolutionize the top-of-the-funnel sales process. In the modern B2B landscape, SDRs are bogged down by manual research and repetitive outreach. Most automation tools are “dumb”—they send the same template to everyone and have zero “memory” of past interactions.

This project solves the Memory Wall in Large Language Models (LLMs). LLMs have a limited context window; if a sales conversation spans weeks, the AI “forgets” earlier details. Our project implements a Hybrid Memory Management System using RAG (Retrieval-Augmented Generation) and Predictive ML to create an agent that researches, remembers, and converts.

The Core Problem & Solution
The Problem
Context Loss: Standard bots treat every email as a new conversation.

High Token Costs: Sending the entire history of a lead to an LLM for every reply is financially unsustainable.

Low Personalization: Without deep research, conversion rates remain below 1%.

The Solution
By building an AI-SDR with Memory Management, we ensure:

Long-term Recall: Using Vector Databases to store “Semantic Memory.”

Cost Efficiency: Using Summarization buffers to keep token counts low.

Lead Intelligence: Using Machine Learning to predict Selection% (which leads are actually worth the effort).

System Architecture & Workflow
The architecture is built on a modular, agentic framework that ensures separation of concerns between data storage, reasoning, and execution.

The Workflow Lifecycle:
Lead Generation (Synthesis): The system generates or ingests a lead profile.

Initial Scoring: The ML Model (Phase 2 feature) analyzes the profile and assigns a “Selection%” score.

Context Retrieval (RAG): The system queries the Vector Database for any historical context regarding the lead’s company or industry.

Research & Personalization: The AI “reasons” through the lead’s pain points.

Drafting: The LLM generates a hyper-personalized email.

Sentiment Feedback Loop: When a lead replies, NLP determines if the sentiment is positive, neutral, or negative.

Memory Pruning: The interaction is summarized and stored back into the Vector DB.

Tech Stack & Domain Expertise
1. Natural Language Processing (NLP)
Libraries: spaCy, NLTK, HuggingFace Transformers.

Function: Used for Named Entity Recognition (NER) to extract data from LinkedIn bios and Intent Classification to understand if a lead is saying “No” or “Maybe later.”

2. Machine Learning (ML)
Libraries: Scikit-Learn, XGBoost, Pandas.

Function: Building the Predictive Lead Scoring model. This model learns from “Synthetic Historical Data” to identify patterns of successful conversions.

3. Retrieval-Augmented Generation (RAG)
Vector Database: Pinecone or ChromaDB.

Embeddings: text-embedding-004 (Google) or text-embedding-3-small (OpenAI).

Logic: Converting text into high-dimensional vectors to allow “Semantic Search.”

4. Databases
Relational (PostgreSQL): Acts as the Pseudo-CRM. Since we don’t have access to Salesforce, we built our own schema to track Lead Status, Metadata, and Timestamps.

NoSQL/Vector: To manage unstructured memory.

5. Backend & Orchestration
Frameworks: FastAPI for the API layer and LangChain/LangGraph for managing agent states.

Frontend: Streamlit for a clean, B.Tech review-ready dashboard.

API Documentation & Integration
The project relies on a multi-API ecosystem to function:

LLM API (Gemini/GPT-4o): The “Brain” that handles reasoning and text generation.

Embedding API: Converts text into mathematical vectors for the RAG system.

Search API (Optional): Integrating tools like Serper.dev to allow the SDR to look up live news about a lead’s company.

📅 Roadmap: The Two-Phase Evolution
Phase 1: The Communicator (MVP)
Goal: Build a functional end-to-end pipeline that can send an email and save a log.

Lead Synthesis: Creating a script to generate realistic data.

Basic Prompting: Establishing the “SDR Persona.”

CRUD Operations: A basic UI to add, view, and delete leads.

Outcome: A system that successfully crafts an outreach message based on a static profile.

Phase 2: The Intelligence (Advanced)
Goal: Implement Memory, ML, and Advanced Analysis for the Research Paper.

Advanced Memory Management: Implementing the Short-term vs. Long-term memory split.

ML Model Training: Creating the Selection% predictor.

Sentiment Analysis: Categorizing lead replies to automate the “Next Best Action.”

Outcome: A strategic agent that prioritizes high-value leads and remembers context from interactions months ago.

Detailed Feature Breakdown
1. Hybrid Memory Management (RAG)
In this project, we address the “context window” issue. When a conversation gets long, we don’t send the whole thing to the LLM.

Short-term: We use a ConversationSummaryBufferMemory. It keeps the last 2-3 messages in raw text but summarizes everything before that into a concise paragraph.

Long-term: Key insights (e.g., “Lead mentioned budget constraints in Q3”) are stored in Pinecone. When the AI prepares a follow-up, it “retrieves” these specific vectors.

2. Predictive Selection% (ML)
This is the “Engineering” heart of the project.

Data Preparation: We synthesize features like industry_growth_rate, lead_seniority, past_interaction_count, and avg_sentiment.

Training: We use an XGBoost Classifier.

Metric: The system outputs a score from 0.0 to 1.0.

Score > 0.8: The AI sends a highly personalized message.

Score < 0.3: The AI puts the lead on a “Waitlist.”

3. Recommendation Engine
The AI doesn’t just reply; it provides a “Recommendation” for the human user.

Analysis: “The lead seems hesitant about price.”

Recommendation: “Offer a 20% early-bird discount or share the ROI Case Study.”

Research Focus: Memory & NLP
As part of our final year B.Tech thesis, we are documenting:

Semantic Retrieval Accuracy: How well does the RAG system find the “right” memory?

Intent Mapping: Using Few-Shot Prompting to improve how the AI categorizes “Sales Objections.”

Memory Decay: Studying the impact of summarization on “Information Loss” in long-term sales cycles.

Installation & Usage
(Detailed steps for cloning the repo, setting up the PostgreSQL database, and configuring the .env file with Gemini/Pinecone keys).

Clone Repository: git clone https://github.com/your-username/ai-sdr

Install Requirements: pip install -r requirements.txt

Initialize Database: python init_db.py

Run Dashboard: streamlit run main.py

Conclusion
The AI-SDR with Memory Management is more than just a chatbot; it is a demonstration of how AI Agents can handle complex, long-term business processes. By splitting the project into a functional MVP (Phase 1) and an intelligent, ML-driven system (Phase 2), we have created a robust platform that addresses real-world sales challenges while meeting all academic requirements for a B.Tech final year project.
