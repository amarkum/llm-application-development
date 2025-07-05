# LLM Application Development - Complete Learning Roadmap

## 📚 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Phase 1: Foundations (Weeks 1-2)](#phase-1-foundations-weeks-1-2)
3. [Phase 2: RAG & Vector Databases (Weeks 3-4)](#phase-2-rag--vector-databases-weeks-3-4)
4. [Phase 3: Production Applications (Weeks 5-6)](#phase-3-production-applications-weeks-5-6)
5. [Phase 4: Advanced Topics (Weeks 7-8)](#phase-4-advanced-topics-weeks-7-8)
6. [Resources & Communities](#resources--communities)
7. [Project Portfolio](#project-portfolio)

---

## Prerequisites

### Required Knowledge
- Python basics (functions, classes, async/await)
- Basic API understanding (REST, JSON)
- Git/GitHub basics

### Setup Your Environment
```bash
# Create a virtual environment
python -m venv llm-env
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate

# Install base packages
pip install langchain openai python-dotenv jupyter notebook
```

### Get API Keys
1. **OpenAI**: [Get API Key](https://platform.openai.com/api-keys)
2. **Anthropic Claude**: [Get API Key](https://console.anthropic.com/)
3. **Hugging Face**: [Get Token](https://huggingface.co/settings/tokens)

---

## Phase 1: Foundations (Weeks 1-2)

### 📺 Video Courses

#### Day 1-3: LLM Basics
1. **LangChain Crash Course** - freeCodeCamp
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=lG7Uxts9SXs)
   - Duration: 1+ hour
   - Topics: Chains, Prompts, Memory, Agents

2. **OpenAI API Tutorial** - Tech With Tim
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=c-g6epk3fFE)
   - Duration: 30 minutes
   - Topics: API basics, Chat completions

### 🛠️ Implementation: Project 1 - Basic Chatbot

```python
# project1_chatbot.py
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def simple_chatbot():
    messages = []
    print("Chatbot started! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        print(f"\nAssistant: {assistant_message}")

if __name__ == "__main__":
    simple_chatbot()
```

#### Day 4-7: LangChain Fundamentals

3. **Build AI Apps with ChatGPT** - Scrimba
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=4qNwoAAfnk4)
   - Duration: 3.5 hours
   - Build complete applications

4. **LangChain Series** - Sam Witteveen
   - 🔗 [Playlist](https://www.youtube.com/playlist?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)
   - Start with first 5 videos

### 🛠️ Implementation: Project 2 - Document Q&A

```python
# project2_document_qa.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_qa_system(file_path):
    # Load document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    return qa

# Usage
qa_system = create_qa_system("document.txt")
response = qa_system.run("What is this document about?")
print(response)
```

---

## Phase 2: RAG & Vector Databases (Weeks 3-4)

### 📺 Video Courses

#### Day 8-10: Vector Databases
1. **Vector Databases Explained** - AssemblyAI
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=klTvEwg3oJ4)
   - Topics: Embeddings, Similarity search

2. **Build a RAG Application** - Sam Witteveen
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=BrsocJb-fAo)
   - Full RAG implementation

### 🛠️ Implementation: Project 3 - Advanced RAG System

```python
# project3_advanced_rag.py
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone

class RAGSystem:
    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.setup_vectorstore()
        
    def setup_vectorstore(self):
        # Load PDFs
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Pinecone
        pinecone.init(
            api_key="YOUR_PINECONE_KEY",
            environment="YOUR_ENV"
        )
        
        # Create vector store
        self.vectorstore = Pinecone.from_documents(
            texts,
            embeddings,
            index_name="pdf-index"
        )
        
    def create_chain(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=self.vectorstore.as_retriever(),
            memory=memory
        )
        
        return chain
```

#### Day 11-14: Production RAG

3. **Production RAG Pipeline** - Data Indy
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=eqOfr4AGLk8)
   - Topics: Chunking strategies, Reranking

4. **LlamaIndex Tutorial** - AssemblyAI
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=kFC-OWw7G8k)
   - Alternative to LangChain

### 🛠️ Implementation: Project 4 - Multi-Source RAG

```python
# project4_multi_source_rag.py
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.document_loaders import WebBaseLoader, GitLoader

class MultiSourceRAG:
    def __init__(self):
        self.documents = []
        
    def load_web_data(self, urls):
        for url in urls:
            loader = WebBaseLoader(url)
            self.documents.extend(loader.load())
            
    def load_github_repo(self, repo_url):
        loader = GitLoader(
            clone_url=repo_url,
            repo_path="./repo",
            branch="main"
        )
        self.documents.extend(loader.load())
        
    def create_hybrid_retriever(self):
        # Create BM25 retriever (keyword search)
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        
        # Create dense retriever (semantic search)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(self.documents, embeddings)
        dense_retriever = vectorstore.as_retriever()
        
        # Ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
```

---

## Phase 3: Production Applications (Weeks 5-6)

### 📺 Video Courses

#### Day 15-17: Full Stack Development
1. **Build a Full Stack AI SaaS** - JavaScript Mastery
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=ffJ38dBzrlY)
   - Duration: 5+ hours
   - Stack: Next.js, Tailwind, Stripe

2. **FastAPI + LangChain** - Patrick Loeber
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=kYRB-vJFy38)
   - Topics: REST APIs, Streaming

### 🛠️ Implementation: Project 5 - Production API

```python
# project5_production_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import redis
import asyncio
from typing import List
import json

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    
class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Check cache
    cache_key = f"chat:{request.session_id}:{request.message}"
    cached_response = redis_client.get(cache_key)
    
    if cached_response:
        return json.loads(cached_response)
    
    # Generate response
    response = await generate_response(request.message, request.session_id)
    
    # Cache response
    redis_client.setex(
        cache_key,
        3600,  # 1 hour TTL
        json.dumps(response)
    )
    
    return response

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def generate():
        async for chunk in generate_stream_response(request.message):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# Deployment script
"""
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.0.340
redis==5.0.1
python-multipart==0.0.6

# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
```

#### Day 18-21: Monitoring & Optimization

3. **LLM Monitoring** - Weights & Biases
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=pHq0NRHL8cg)
   - Topics: Cost tracking, Performance monitoring

4. **Caching Strategies** - Nicholas Renotte
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=vEZHgkltM_8)
   - Topics: Redis, Response caching

### 🛠️ Implementation: Project 6 - Complete SaaS Backend

```python
# project6_saas_backend.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import stripe
from datetime import datetime
import jwt

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    stripe_customer_id = Column(String)
    api_calls = Column(Integer, default=0)
    subscription_tier = Column(String, default="free")

class LLMSaaS:
    def __init__(self):
        self.app = FastAPI()
        self.setup_database()
        self.setup_routes()
        
    def setup_database(self):
        engine = create_engine("postgresql://user:pass@localhost/db")
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)
        
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    def check_rate_limit(self, user: User):
        limits = {
            "free": 10,
            "pro": 1000,
            "enterprise": 10000
        }
        
        if user.api_calls >= limits.get(user.subscription_tier, 10):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
    def track_usage(self, user: User, tokens: int, cost: float):
        # Log to database
        usage = Usage(
            user_id=user.id,
            tokens=tokens,
            cost=cost,
            timestamp=datetime.utcnow()
        )
        db.add(usage)
        db.commit()
        
        # Send to monitoring service
        wandb.log({
            "user_id": user.id,
            "tokens": tokens,
            "cost": cost
        })
```

---

## Phase 4: Advanced Topics (Weeks 7-8)

### 📺 Video Courses

#### Day 22-24: Agents & Tools
1. **Building AI Agents** - Matt Wolfe
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=ziu87EXZVUE)
   - Topics: ReAct pattern, Tool usage

2. **AutoGPT Tutorial** - Dave Ebbelaar
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=L6tZV2Qn3xc)
   - Topics: Autonomous agents

### 🛠️ Implementation: Project 7 - AI Agent System

```python
# project7_ai_agent.py
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
import requests

class CustomAgent:
    def __init__(self):
        self.tools = self.setup_tools()
        self.agent = self.create_agent()
        
    def setup_tools(self):
        # Search tool
        search = DuckDuckGoSearchRun()
        
        # Wikipedia tool
        wikipedia = WikipediaAPIWrapper()
        
        # Custom API tool
        def call_api(query: str) -> str:
            response = requests.get(f"https://api.example.com/data?q={query}")
            return response.json()
        
        # Code execution tool
        def execute_python(code: str) -> str:
            try:
                exec_globals = {}
                exec(code, exec_globals)
                return str(exec_globals.get('result', 'No result'))
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Search the internet for information"
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Get information from Wikipedia"
            ),
            Tool(
                name="API",
                func=call_api,
                description="Call external API for data"
            ),
            Tool(
                name="Python",
                func=execute_python,
                description="Execute Python code"
            )
        ]
        
        return tools
        
    def create_agent(self):
        # Custom prompt template
        template = """You are an AI assistant with access to multiple tools.
        
        Tools available:
        {tools}
        
        Use this format:
        Thought: Consider what to do
        Action: the action to take
        Action Input: the input to the action
        Observation: the result
        ... (repeat as needed)
        Thought: I have the final answer
        Final Answer: the final answer
        
        Question: {input}
        {agent_scratchpad}"""
        
        prompt = StringPromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"]
        )
        
        # Create agent
        llm_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
        
        return agent_executor
```

#### Day 25-28: Safety & Compliance

3. **LLM Guardrails** - Guardrails AI
   - 🔗 [Watch Video](https://www.youtube.com/watch?v=6n6t65F6vwU)
   - Topics: Output validation, Safety checks

4. **Production Best Practices** - DeepLearning.AI
   - 🔗 [Course Link](https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/)
   - Free course on production patterns

### 🛠️ Implementation: Project 8 - Safety System

```python
# project8_safety_system.py
from guardrails import Guard
from guardrails.hub import ToxicLanguage, PiiMasking, FactualityCheck
import logging
from typing import Dict, Any
import hashlib

class LLMSafetySystem:
    def __init__(self):
        self.setup_guards()
        self.setup_logging()
        
    def setup_guards(self):
        # Toxicity guard
        self.toxicity_guard = Guard().use(
            ToxicLanguage,
            threshold=0.5,
            validation_method="sentence"
        )
        
        # PII guard
        self.pii_guard = Guard().use(
            PiiMasking,
            pii_types=["email", "phone", "ssn", "credit_card"]
        )
        
        # Custom business rules
        self.business_guard = Guard.from_string(
            validators=[
                self.check_financial_advice,
                self.check_medical_advice,
                self.check_legal_advice
            ]
        )
        
    def check_financial_advice(self, value: str) -> Dict[str, Any]:
        keywords = ["invest", "stock", "trading", "financial advice"]
        if any(keyword in value.lower() for keyword in keywords):
            return {
                "valid": False,
                "reason": "Financial advice detected",
                "fix": "Add disclaimer: 'This is not financial advice'"
            }
        return {"valid": True}
        
    def validate_input(self, user_input: str) -> Dict[str, Any]:
        # Check for prompt injection
        injection_patterns = [
            "ignore all previous",
            "disregard instructions",
            "new system prompt"
        ]
        
        for pattern in injection_patterns:
            if pattern in user_input.lower():
                return {
                    "safe": False,
                    "reason": "Potential prompt injection"
                }
                
        return {"safe": True}
        
    def validate_output(self, llm_output: str) -> Dict[str, Any]:
        # Run through all guards
        try:
            # Toxicity check
            self.toxicity_guard.validate(llm_output)
            
            # PII check and masking
            masked_output = self.pii_guard.validate(llm_output)
            
            # Business rules
            self.business_guard.validate(masked_output)
            
            return {
                "safe": True,
                "output": masked_output,
                "modifications": ["PII masked"] if masked_output != llm_output else []
            }
            
        except Exception as e:
            return {
                "safe": False,
                "reason": str(e),
                "fallback": "I cannot provide that information."
            }
            
    def log_interaction(self, user_id: str, input_text: str, output_text: str):
        # Create audit log
        timestamp = datetime.utcnow()
        interaction_hash = hashlib.sha256(
            f"{user_id}{input_text}{output_text}{timestamp}".encode()
        ).hexdigest()
        
        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "input_hash": hashlib.sha256(input_text.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_text.encode()).hexdigest(),
            "interaction_hash": interaction_hash,
            "compliance_checks": ["toxicity", "pii", "business_rules"]
        }
        
        # Store in append-only log
        logging.info(f"AUDIT: {json.dumps(log_entry)}")
```

---

## Resources & Communities

### 📚 Documentation
- **LangChain Docs**: [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
- **OpenAI Cookbook**: [https://cookbook.openai.com/](https://cookbook.openai.com/)
- **Hugging Face Docs**: [https://huggingface.co/docs](https://huggingface.co/docs)

### 💬 Communities
- **Discord Servers**:
  - LangChain: [Join Discord](https://discord.gg/langchain)
  - Hugging Face: [Join Discord](https://discord.gg/huggingface)
  - OpenAI: [Join Discord](https://discord.gg/openai)

- **Reddit**:
  - r/LocalLLaMA: [Visit Subreddit](https://reddit.com/r/LocalLLaMA)
  - r/OpenAI: [Visit Subreddit](https://reddit.com/r/OpenAI)
  - r/LangChain: [Visit Subreddit](https://reddit.com/r/LangChain)

### 🛠️ Tools & Platforms
- **Vector Databases**:
  - Pinecone: [https://www.pinecone.io/](https://www.pinecone.io/)
  - ChromaDB: [https://www.trychroma.com/](https://www.trychroma.com/)
  - Weaviate: [https://weaviate.io/](https://weaviate.io/)

- **Monitoring**:
  - Langfuse: [https://langfuse.com/](https://langfuse.com/)
  - Helicone: [https://www.helicone.ai/](https://www.helicone.ai/)
  - Weights & Biases: [https://wandb.ai/](https://wandb.ai/)

### 📖 Books & Courses
- **Books**:
  - "Building LLM Apps" by Sowmya Vajjala
  - "Hands-On Large Language Models" by Jay Alammar

- **Paid Courses**:
  - DeepLearning.AI LLM Specialization
  - Fast.AI Practical Deep Learning

---

## Project Portfolio

### Build These Projects in Order:

1. **Week 1-2**: Basic Chatbot
   - Simple Q&A bot
   - Add memory
   - Deploy to Streamlit

2. **Week 3-4**: Document Analysis System
   - PDF loader
   - Semantic search
   - Source citations

3. **Week 5-6**: Full Stack AI App
   - User authentication
   - Payment integration
   - Usage tracking

4. **Week 7-8**: Production Agent
   - Multi-tool agent
   - Safety guardrails
   - Compliance logging

### GitHub Portfolio Structure:
```
amarkum/
├── llm-application-development/
│   ├── 01-basic-chatbot/
│   ├── 02-document-qa/
│   ├── 03-rag-system/
│   ├── 04-production-api/
│   ├── 05-saas-backend/
│   ├── 06-ai-agent/
│   ├── 07-safety-system/
│   └── README.md
```

### Deployment Checklist:
- [ ] Environment variables secured
- [ ] Rate limiting implemented
- [ ] Error handling complete
- [ ] Logging configured
- [ ] Tests written
- [ ] Documentation complete
- [ ] CI/CD pipeline setup
- [ ] Monitoring enabled

---

## Next Steps

1. **Start Today**: Watch the first video and code along
2. **Join Communities**: Get help when stuck
3. **Build Projects**: One small project per week
4. **Share Progress**: Post on LinkedIn/Twitter
5. **Apply for Jobs**: After completing 4-5 projects

Remember: The key to learning is building. Don't just watch videos - implement the code, break things, and fix them. Good luck on your LLM development journey! 🚀
