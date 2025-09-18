import os
import json
import logging
import html
import uuid
import re
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from kubernetes import client, config
from typing import TypedDict, List, Optional
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from cachetools import TTLCache
from dotenv import load_dotenv
from datetime import datetime, timezone

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session cache with a 5-minute (300 seconds) TTL
session_cache = TTLCache(maxsize=1024, ttl=300)

# --- Helper Functions ---
def get_relative_age(timestamp):
    """Converts a datetime object to a human-readable relative age string."""
    if not timestamp:
        return "N/A"
    now = datetime.now(timezone.utc)
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = now - timestamp
    
    if age.days > 0:
        return f"{age.days}d"
    elif age.seconds >= 3600:
        return f"{age.seconds // 3600}h"
    elif age.seconds >= 60:
        return f"{age.seconds // 60}m"
    else:
        return f"{age.seconds}s"

# --- Kubernetes Configuration ---
try:
    config.load_incluster_config()
    logger.info("Loaded in-cluster Kubernetes config.")
except config.ConfigException:
    try:
        config.load_kube_config()
        logger.info("Loaded local Kubernetes config.")
    except config.ConfigException:
        logger.error("Could not load any Kubernetes configuration.")
        exit(1)

k8s_core_v1 = client.CoreV1Api()
k8s_apps_v1 = client.AppsV1Api()

# --- Bedrock LLM Configuration ---
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    )
    logger.info("Bedrock client created successfully.")
except (NoCredentialsError, PartialCredentialsError) as e:
    logger.error(f"AWS credentials not found. Error: {e}")
    llm = None

# --- FastAPI App ---
app = FastAPI()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")

# --- LangGraph State (with Memory) ---
class GraphState(TypedDict):
    user_message: str
    chat_history: List[BaseMessage]
    intent: dict
    k8s_result: dict # Can now hold structured data (logs, events)
    final_response: str

# --- LangGraph Nodes (Updated for Conversational Context) ---

def nlu_agent(state: GraphState):
    """Converts natural language to JSON intent, using conversation history for context."""
    logger.info("Executing NLU Agent")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are an expert at converting natural language to a JSON intent for a Kubernetes chat bot.
                Use the conversation history to understand context and resolve references.
                Your output must be a single, valid JSON object.
                Valid actions are: "get_pods", "get_services", "get_logs", "scale", "diagnose_pod".

                --- CONVERSATION HISTORY ---
                {chat_history}
                
                --- EXAMPLES ---
                User: "show me pods" -> {{ "action": "get_pods" }}
                User: "list all services" -> {{ "action": "get_services" }}
                User: "scale frontend to 3 replicas" -> {{ "action": "scale", "resource": "deployment", "name": "frontend", "replicas": 3 }}
                User: "get logs for backend-abc" -> {{ "action": "get_logs", "pod": "backend-abc" }}
                
                --- CONTEXTUAL EXAMPLES ---
                (History shows a pod named 'backend-xyz-123' is in CrashLoopBackOff)
                User: "why is that one failing?" -> {{ "action": "diagnose_pod", "pod": "backend-xyz-123" }}
                User: "get logs for that pod" -> {{ "action": "get_logs", "pod": "backend-xyz-123" }}
                '''
            ),
            ("human", "{user_message}"),
        ]
    )
    if not llm:
        raise RuntimeError("LLM is not initialized.")

    chain = prompt | llm
    response = chain.invoke({"user_message": state["user_message"], "chat_history": state["chat_history"]})
    
    try:
        # Use regex for better JSON extraction
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.content)
        if json_match:
            json_str = json_match.group()
            intent = json.loads(json_str)
            logger.info(f"NLU generated intent: {intent}")
            return {"intent": intent}
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error decoding LLM response into JSON: {e}")
        return {"intent": {"action": "error", "details": "I couldn't understand that request."}}}

def validator_agent(state: GraphState):
    """Validates the intent from the NLU agent."""
    logger.info("Executing Validator Agent")
    intent = state.get("intent", {})
    action = intent.get("action")
    
    if not action or action not in ["get_pods", "get_services", "get_logs", "scale", "diagnose_pod"]:
        intent["action"] = "error"
        intent["details"] = "Invalid action specified."
    
    # Validate required parameters
    if action == "get_logs" and not intent.get("pod"):
        intent["action"] = "error"
        intent["details"] = "Pod name is required for getting logs."
    elif action == "scale" and (not intent.get("name") or not intent.get("replicas")):
        intent["action"] = "error"
        intent["details"] = "Deployment name and replica count are required for scaling."
    elif action == "diagnose_pod" and not intent.get("pod"):
        intent["action"] = "error"
        intent["details"] = "Pod name is required for diagnosis."
    
    return {"intent": intent}

def k8s_executor(state: GraphState):
    """Executes the validated Kubernetes action."""
    logger.info("Executing K8s Executor")
    intent = state["intent"]
    action = intent["action"]
    result = {}

    try:
        if action == "get_pods":
            pods = k8s_core_v1.list_namespaced_pod(namespace="default", watch=False)
            pod_list = []
            for i in pods.items:
                ready_containers = sum([1 for c in i.status.container_statuses if c.ready]) if i.status.container_statuses else 0
                total_containers = len(i.spec.containers)
                restarts = sum([c.restart_count for c in i.status.container_statuses]) if i.status.container_statuses else 0
                age = get_relative_age(i.status.start_time)
                pod_list.append(f'{i.metadata.name},{ready_containers}/{total_containers},{i.status.phase},{restarts},{age},{i.status.pod_ip},{i.spec.node_name}')
            result["raw"] = "NAME,READY,STATUS,RESTARTS,AGE,IP,NODE\n" + "\n".join(pod_list)

        elif action == "get_services":
            services = k8s_core_v1.list_namespaced_service(namespace="default", watch=False)
            service_list = []
            for i in services.items:
                ports = ",".join([f'{p.port}:{p.node_port}/{p.protocol}' for p in i.spec.ports]) if i.spec.type == "NodePort" else ",".join([f'{p.port}/{p.protocol}' for p in i.spec.ports])
                age = get_relative_age(i.metadata.creation_timestamp)
                service_list.append(f'{i.metadata.name},{i.spec.cluster_ip},{i.spec.external_i_ps if i.spec.external_i_ps else '<none>'},{ports},{age}')
            result["raw"] = "NAME,TYPE,CLUSTER-IP,EXTERNAL-IP,PORT(S),AGE\n" + "\n".join(service_list)

        elif action == "get_logs":
            if "pod" not in intent:
                result["raw"] = "Error: Pod name is required."
            else:
                result["raw"] = k8s_core_v1.read_namespaced_pod_log(name=intent["pod"], namespace="default")

        elif action == "scale":
            if "name" not in intent or "replicas" not in intent:
                result["raw"] = "Error: Deployment name and replica count are required."
            else:
                k8s_apps_v1.patch_namespaced_deployment_scale(name=intent["name"], namespace="default", body={"spec": {"replicas": intent["replicas"]}})
                result["raw"] = f"Deployment {intent['name']} scaled to {intent['replicas']} replicas."

        elif action == "diagnose_pod":
            if "pod" not in intent:
                result["raw"] = "Error: Pod name is required."
            else:
                pod_name = intent["pod"]
                logs = k8s_core_v1.read_namespaced_pod_log(name=pod_name, namespace="default", tail_lines=50)
                pod_info = k8s_core_v1.read_namespaced_pod(name=pod_name, namespace="default")
                events = k8s_core_v1.list_namespaced_event(namespace="default", field_selector=f'involvedObject.name={pod_name}')
                result["logs"] = logs
                result["description"] = pod_info.to_dict() # More efficient serialization
                result["events"] = "\n".join([f'{e.last_timestamp} {e.type} {e.reason}: {e.message}' for e in events.items])

        elif action == "error":
            result["raw"] = intent.get("details", "An unknown error occurred.")

    except client.ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        result["raw"] = f"Error executing Kubernetes command: {e.reason}"
    except Exception as e:
        logger.error(f"An unexpected error in k8s_executor: {e}")
        result["raw"] = "An unexpected error occurred."

    return {"k8s_result": result}

def responder_agent(state: GraphState):
    """Formats the result into a user-friendly response, using an LLM for synthesis if needed."""
    logger.info("Executing Responder Agent")
    k8s_result = state["k8s_result"]
    intent = state["intent"]
    final_response = ""

    if intent["action"] == "diagnose_pod":
        if not llm:
            final_response = "Error: LLM not available for diagnosis."
        else:
            logger.info("Responder using LLM to synthesize diagnosis.")
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant specializing in Kubernetes. Your task is to analyze pod logs, descriptions, and events to explain why a pod is failing in simple terms."),
                ("human", "Please analyze the following data for pod '{pod_name}' and provide a summary of the problem.\n\n---LOGS (last 50 lines)---\n{logs}\n\n---EVENTS---\n{events}\n\n---DESCRIPTION (YAML)---\n{description}"),
            ])
            chain = synthesis_prompt | llm
            response = chain.invoke({
                "pod_name": intent["pod"],
                "logs": k8s_result.get("logs", "Not available."),
                "events": k8s_result.get("events", "Not available."),
                "description": str(k8s_result.get("description", "Not available."))
            })
            final_response = html.escape(response.content)

    elif intent["action"] in ["get_pods", "get_services"]:
        # Convert CSV-like string to markdown table with proper escaping
        lines = k8s_result.get("raw", "").strip().split('\n')
        if not lines or not lines[0]:
            final_response = "No resources found."
        else:
            import csv
            from io import StringIO
            
            # Use proper CSV parsing
            csv_data = StringIO(k8s_result.get("raw", ""))
            reader = csv.reader(csv_data)
            rows = list(reader)
            
            if rows:
                header = [html.escape(cell) for cell in rows[0]]
                table_rows = []
                for row in rows[1:]:
                    escaped_row = [html.escape(cell) for cell in row]
                    table_rows.append(f"| {' | '.join(escaped_row)} |")
                
                markdown_table = f"| {' | '.join(header)} |\n|{'|'.join(['---'] * len(header))}|\n"
                markdown_table += "\n".join(table_rows)
                final_response = markdown_table
            else:
                final_response = "No resources found."
    else:
        # For logs, wrap in markdown code block with escaping
        raw_result = k8s_result.get("raw", "No result found.")
        if intent["action"] == "get_logs":
            final_response = f"```\n{html.escape(raw_result)}\n```"
        else:
            final_response = html.escape(raw_result)

    return {"final_response": final_response}

# --- LangGraph Workflow Definition ---
workflow = StateGraph(GraphState)
workflow.add_node("nlu", nlu_agent)
workflow.add_node("validator", validator_agent)
workflow.add_node("executor", k8s_executor)
workflow.add_node("responder", responder_agent)

workflow.set_entry_point("nlu")
workflow.add_edge("nlu", "validator")
workflow.add_edge("validator", "executor")
workflow.add_edge("executor", "responder")
workflow.add_edge("responder", END)

app_graph = workflow.compile()

# --- API Endpoints (Updated for Session Management) ---
@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest, x_session_id: Optional[str] = Header(None)):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not configured.")

    # Use session header or generate UUID-based session ID
    session_id = x_session_id or str(uuid.uuid4())
    
    # Retrieve history from cache or start a new one
    chat_history = session_cache.get(session_id, [])
    
    # Sanitize user input
    sanitized_message = html.escape(chat_request.message.strip())
    
    inputs = {"user_message": sanitized_message, "chat_history": chat_history}
    
    try:
        result = app_graph.invoke(inputs)
        final_response = result["final_response"]
        
        # Update history and save back to cache
        updated_history = chat_history + [HumanMessage(content=sanitized_message), AIMessage(content=final_response)]
        session_cache[session_id] = updated_history
        
        return {"reply": final_response, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
