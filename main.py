import os
import json
import logging
import html
import uuid
import re
import subprocess
import yaml
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

# Cluster context cache
cluster_context_cache = {}

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
    message: str = Field(..., min_length=1, max_length=2000, description="User message")

# --- LangGraph State (with Memory) ---
class GraphState(TypedDict):
    user_message: str
    chat_history: List[BaseMessage]
    cluster_context: str
    kubectl_command: str
    command_output: str
    final_response: str

# --- LangGraph Nodes (Updated for Conversational Context) ---

def get_cluster_context():
    """Get comprehensive cluster context for the AI model."""
    try:
        context = {}
        
        # Get pods
        pods = k8s_core_v1.list_pod_for_all_namespaces()
        context['pods'] = [{
            'name': p.metadata.name,
            'namespace': p.metadata.namespace,
            'status': p.status.phase,
            'ready': sum([1 for c in p.status.container_statuses if c.ready]) if p.status.container_statuses else 0,
            'total': len(p.spec.containers),
            'restarts': sum([c.restart_count for c in p.status.container_statuses]) if p.status.container_statuses else 0,
            'node': p.spec.node_name
        } for p in pods.items[:20]]  # Limit to 20 pods
        
        # Get services
        services = k8s_core_v1.list_service_for_all_namespaces()
        context['services'] = [{
            'name': s.metadata.name,
            'namespace': s.metadata.namespace,
            'type': s.spec.type,
            'cluster_ip': s.spec.cluster_ip
        } for s in services.items[:10]]  # Limit to 10 services
        
        # Get deployments
        deployments = k8s_apps_v1.list_deployment_for_all_namespaces()
        context['deployments'] = [{
            'name': d.metadata.name,
            'namespace': d.metadata.namespace,
            'replicas': d.spec.replicas,
            'ready_replicas': d.status.ready_replicas or 0
        } for d in deployments.items[:10]]  # Limit to 10 deployments
        
        # Get nodes
        nodes = k8s_core_v1.list_node()
        context['nodes'] = [{
            'name': n.metadata.name,
            'status': 'Ready' if any(c.type == 'Ready' and c.status == 'True' for c in n.status.conditions) else 'NotReady'
        } for n in nodes.items]
        
        return yaml.dump(context, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error getting cluster context: {e}")
        return "Error getting cluster context"

def context_agent(state: GraphState):
    """Gathers current cluster context for the AI model."""
    logger.info("Gathering cluster context")
    cluster_context = get_cluster_context()
    return {"cluster_context": cluster_context}

def command_generator_agent(state: GraphState):
    """AI agent that generates kubectl commands based on user request and cluster context."""
    logger.info("Generating kubectl command")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an expert Kubernetes administrator AI. Based on the user's request and current cluster context, generate the appropriate kubectl command.

RULES:
1. ONLY output the kubectl command, nothing else
2. Use proper kubectl syntax
3. Consider the current cluster state
4. For dangerous operations, add --dry-run=client first
5. Use appropriate namespaces based on context
6. Be precise with resource names from the context

EXAMPLES:
- "show all pods" -> kubectl get pods --all-namespaces
- "delete pod xyz" -> kubectl delete pod xyz
- "scale deployment abc to 5" -> kubectl scale deployment abc --replicas=5
- "get logs from pod xyz" -> kubectl logs xyz
- "describe node master" -> kubectl describe node master
- "create a nginx deployment" -> kubectl create deployment nginx --image=nginx
- "apply this yaml file" -> kubectl apply -f filename.yaml

CURRENT CLUSTER CONTEXT:
{cluster_context}

CONVERSATION HISTORY:
{chat_history}'''),
        ("human", "{user_message}")
    ])
    
    if not llm:
        return {"kubectl_command": "echo 'LLM not available'"}
    
    chain = prompt | llm
    response = chain.invoke({
        "user_message": state["user_message"],
        "cluster_context": state["cluster_context"],
        "chat_history": state["chat_history"]
    })
    
    # Extract kubectl command from response
    command = response.content.strip()
    # Remove any markdown formatting
    command = re.sub(r'```.*?\n', '', command)
    command = re.sub(r'\n```', '', command)
    command = command.strip()
    
    # Ensure it starts with kubectl
    if not command.startswith('kubectl'):
        command = f"kubectl {command}"
    
    logger.info(f"Generated command: {command}")
    return {"kubectl_command": command}

def command_executor_agent(state: GraphState):
    """Executes the kubectl command safely."""
    logger.info("Executing kubectl command")
    command = state["kubectl_command"]
    
    # Safety checks
    dangerous_commands = ['delete', 'rm', 'destroy']
    if any(danger in command.lower() for danger in dangerous_commands):
        if '--dry-run=client' not in command and '--force' not in command:
            command += ' --dry-run=client'
    
    try:
        # Execute kubectl command
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout
        else:
            output = f"Error: {result.stderr}"
        
        logger.info(f"Command executed: {command}")
        logger.info(f"Output length: {len(output)}")
        
        return {"command_output": output}
        
    except subprocess.TimeoutExpired:
        return {"command_output": "Command timed out after 30 seconds"}
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return {"command_output": f"Error executing command: {str(e)}"}



def response_formatter_agent(state: GraphState):
    """Formats the command output into a user-friendly response."""
    logger.info("Formatting response")
    
    command = state["kubectl_command"]
    output = state["command_output"]
    
    # Format the response
    if output.startswith("Error:"):
        final_response = f"❌ **Command Failed**\n\n`{command}`\n\n```\n{output}\n```"
    else:
        # Check if output looks like a table
        lines = output.strip().split('\n')
        if len(lines) > 1 and any(char in lines[0] for char in ['NAME', 'NAMESPACE', 'STATUS']):
            # Format as table
            try:
                header = lines[0].split()
                rows = []
                for line in lines[1:]:
                    if line.strip():
                        rows.append(line.split())
                
                if rows:
                    # Create markdown table
                    table = f"| {' | '.join(header)} |\n"
                    table += f"|{'|'.join(['---'] * len(header))}|\n"
                    for row in rows:
                        # Pad row to match header length
                        while len(row) < len(header):
                            row.append('')
                        table += f"| {' | '.join(row[:len(header)])} |\n"
                    
                    final_response = f"✅ **Command Executed**\n\n`{command}`\n\n{table}"
                else:
                    final_response = f"✅ **Command Executed**\n\n`{command}`\n\n```\n{output}\n```"
            except:
                final_response = f"✅ **Command Executed**\n\n`{command}`\n\n```\n{output}\n```"
        else:
            # Format as code block
            final_response = f"✅ **Command Executed**\n\n`{command}`\n\n```\n{output}\n```"
    
    return {"final_response": final_response}

# --- LangGraph Workflow Definition ---
workflow = StateGraph(GraphState)
workflow.add_node("context", context_agent)
workflow.add_node("command_gen", command_generator_agent)
workflow.add_node("executor", command_executor_agent)
workflow.add_node("formatter", response_formatter_agent)

workflow.set_entry_point("context")
workflow.add_edge("context", "command_gen")
workflow.add_edge("command_gen", "executor")
workflow.add_edge("executor", "formatter")
workflow.add_edge("formatter", END)

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
    
    inputs = {
        "user_message": sanitized_message, 
        "chat_history": chat_history,
        "cluster_context": "",
        "kubectl_command": "",
        "command_output": "",
        "final_response": ""
    }
    
    try:
        result = app_graph.invoke(inputs)
        final_response = result["final_response"]
        
        # Update history and save back to cache
        updated_history = chat_history + [HumanMessage(content=sanitized_message), AIMessage(content=final_response)]
        session_cache[session_id] = updated_history
        
        return {
            "reply": final_response, 
            "session_id": session_id,
            "command": result.get("kubectl_command", ""),
            "raw_output": result.get("command_output", "")
        }
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
