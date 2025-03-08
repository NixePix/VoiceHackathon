from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
from typing import List

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ConversationRequest(BaseModel):
    messages: List[Message]
    agent_id: str
    api_key: str


class RAGRequest(BaseModel):
    document_id: str
    agent_id: str
    api_key: str
    embedding_model: str = "e5_mistral_7b_instruct"
    max_documents_length: int = 10000


async def process_rag(request: RAGRequest):
    try:
        # First, index the document for RAG
        index_url = (
            "https://api.elevenlabs.io/v1/conversational-ai/"
            f"knowledge-base/{request.document_id}/rag-index"
        )
        index_response = requests.post(
            index_url,
            headers={"xi-api-key": request.api_key},
            json={"model": request.embedding_model},
        )

        if index_response.status_code != 200:
            raise HTTPException(
                status_code=index_response.status_code,
                detail="Failed to index document",
            )

        # Check indexing status
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            status_response = requests.get(
                index_url, headers={"xi-api-key": request.api_key}
            )

            if status_response.status_code != 200:
                raise HTTPException(
                    status_code=status_response.status_code,
                    detail="Failed to check indexing status",
                )

            status = status_response.json().get("status")
            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise HTTPException(status_code=400, detail="Failed to index document")

            retry_count += 1
            await time.sleep(5)  # Wait 5 seconds before checking again

        if retry_count >= max_retries:
            raise HTTPException(
                status_code=408, detail="Timeout waiting for document indexing"
            )

        # Get current agent configuration
        agent_url = (
            "https://api.elevenlabs.io/v1/conversational-ai/"
            f"agents/{request.agent_id}"
        )
        agent_response = requests.get(
            agent_url, headers={"xi-api-key": request.api_key}
        )

        if agent_response.status_code != 200:
            raise HTTPException(
                status_code=agent_response.status_code,
                detail="Failed to get agent configuration",
            )

        agent_config = agent_response.json()

        # Enable RAG in the agent configuration
        agent_config["agent"]["prompt"]["rag"] = {
            "enabled": True,
            "embedding_model": request.embedding_model,
            "max_documents_length": request.max_documents_length,
        }

        # Update document usage mode
        kb = agent_config["agent"]["prompt"]["knowledge_base"]
        for i, doc in enumerate(kb):
            if doc["id"] == request.document_id:
                kb[i]["usage_mode"] = "auto"

        # Update the agent configuration
        headers = {"xi-api-key": request.api_key}
        update_response = requests.put(agent_url, headers=headers, json=agent_config)

        if update_response.status_code != 200:
            raise HTTPException(
                status_code=update_response.status_code,
                detail="Failed to update agent configuration",
            )

        return {
            "status": "success",
            "message": "RAG configuration completed successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/talk")
async def handle_conversation(request: ConversationRequest):
    try:
        # Endpoint for conversation with the agent
        conversation_url = (
            "https://api.elevenlabs.io/v1/conversational-ai/"
            f"agents/{request.agent_id}/conversation"
        )

        # Send conversation to ElevenLabs API
        conversation_response = requests.post(
            conversation_url,
            headers={"xi-api-key": request.api_key},
            json={"messages": [msg.dict() for msg in request.messages]},
        )

        if conversation_response.status_code != 200:
            raise HTTPException(
                status_code=conversation_response.status_code,
                detail="Failed to process conversation",
            )

        # Get conversation response
        conversation_data = conversation_response.json()

        # Check if this is the last message in conversation
        last_message = request.messages[-1]
        if last_message.content.lower() in ["goodbye", "bye", "end"]:
            # Extract document ID from conversation context
            doc_id = conversation_data.get("document_id")
            if doc_id:
                # Process RAG after conversation ends
                rag_request = RAGRequest(
                    document_id=doc_id,
                    agent_id=request.agent_id,
                    api_key=request.api_key,
                )
                await process_rag(rag_request)
                conversation_data["rag_status"] = "processed"

        return conversation_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
