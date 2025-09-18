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
    current_step: int
    max_steps: int
    command_history: List[dict]
    current_action: str
    current_output: str
    analysis_result: str
    next_action_needed: bool
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

def command_planner_agent(state: GraphState):
    """AI agent that plans the next action based on user request and previous results."""
    logger.info(f"Planning action - Step {state.get('current_step', 1)}")
    
    # Initialize state if first step
    if state.get("current_step", 0) == 0:
        return {
            "current_step": 1,
            "max_steps": 5,
            "command_history": [],
            "next_action_needed": True
        }
    
    command_history_str = "\n".join([
        f"Step {i+1}: {cmd['action']} -> {cmd['output'][:200]}..."
        for i, cmd in enumerate(state.get("command_history", []))
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an expert Kubernetes administrator AI that can chain multiple commands to solve complex problems.

Based on the user's request and previous command results, determine the next action.

Output ONLY a JSON object:
- If more investigation needed: {"action": "action_name", "params": {...}, "continue": true, "reasoning": "why this action"}
- If task complete: {"action": "complete", "continue": false, "summary": "final summary"}

AVAILABLE ACTIONS:
- get_pods, get_services, get_deployments, get_nodes
- describe_pod, get_logs, get_events
- get_replicasets, get_configmaps, get_secrets
- analyze_resource_usage, check_pod_dependencies

USER REQUEST: {user_message}

CLUSTER CONTEXT:
{cluster_context}

PREVIOUS COMMANDS:
{command_history}

CURRENT STEP: {current_step}/{max_steps}'''),
        ("human", "What should I do next to fully answer the user's request?")
    ])
    
    if not llm:
        return {"current_action": '{"action": "get_pods", "namespace": "all"}'}
    
    chain = prompt | llm
    response = chain.invoke({
        "user_message": state["user_message"],
        "cluster_context": state["cluster_context"],
        "command_history": command_history_str,
        "current_step": state.get("current_step", 1),
        "max_steps": state.get("max_steps", 5)
    })
    
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.content)
        if json_match:
            action_data = json.loads(json_match.group())
            logger.info(f"Planned action: {action_data}")
            
            if action_data.get("continue", True) and state.get("current_step", 1) < state.get("max_steps", 5):
                return {
                    "current_action": json.dumps(action_data),
                    "next_action_needed": True
                }
            else:
                return {
                    "current_action": json.dumps(action_data),
                    "next_action_needed": False
                }
        else:
            return {"current_action": '{"action": "complete", "continue": false}'}
    except Exception as e:
        logger.error(f"Error parsing action: {e}")
        return {"current_action": '{"action": "complete", "continue": false}'}

def action_executor_agent(state: GraphState):
    """Executes Kubernetes operations using the Python client."""
    logger.info("Executing Kubernetes operation")
    
    try:
        action_data = json.loads(state["current_action"])
        action = action_data.get("action")
        
        if action == "complete":
            return {"current_output": "Task completed"}
        
        if action == "get_pods":
            namespace = action_data.get("namespace", "all")
            if namespace == "all":
                pods = k8s_core_v1.list_pod_for_all_namespaces()
            else:
                pods = k8s_core_v1.list_namespaced_pod(namespace=namespace)
            
            output = "NAME\tNAMESPACE\tSTATUS\tREADY\tRESTARTS\tAGE\n"
            for pod in pods.items:
                ready = sum([1 for c in pod.status.container_statuses if c.ready]) if pod.status.container_statuses else 0
                total = len(pod.spec.containers)
                restarts = sum([c.restart_count for c in pod.status.container_statuses]) if pod.status.container_statuses else 0
                age = get_relative_age(pod.status.start_time)
                output += f"{pod.metadata.name}\t{pod.metadata.namespace}\t{pod.status.phase}\t{ready}/{total}\t{restarts}\t{age}\n"
        
        elif action == "get_services":
            namespace = action_data.get("namespace", "all")
            if namespace == "all":
                services = k8s_core_v1.list_service_for_all_namespaces()
            else:
                services = k8s_core_v1.list_namespaced_service(namespace=namespace)
            
            output = "NAME\tNAMESPACE\tTYPE\tCLUSTER-IP\tPORTS\n"
            for svc in services.items:
                ports = ",".join([f"{p.port}/{p.protocol}" for p in svc.spec.ports]) if svc.spec.ports else "None"
                output += f"{svc.metadata.name}\t{svc.metadata.namespace}\t{svc.spec.type}\t{svc.spec.cluster_ip}\t{ports}\n"
        
        elif action == "get_logs":
            pod_name = action_data.get("pod")
            namespace = action_data.get("namespace", "default")
            if pod_name:
                output = k8s_core_v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=100)
            else:
                output = "Error: Pod name required"
        
        elif action == "describe_pod":
            pod_name = action_data.get("pod")
            namespace = action_data.get("namespace", "default")
            if pod_name:
                pod = k8s_core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                output = f"Name: {pod.metadata.name}\n"
                output += f"Namespace: {pod.metadata.namespace}\n"
                output += f"Status: {pod.status.phase}\n"
                output += f"Node: {pod.spec.node_name}\n"
                output += f"IP: {pod.status.pod_ip}\n"
                if pod.spec.containers:
                    output += f"Image: {pod.spec.containers[0].image}\n"
            else:
                output = "Error: Pod name required"
        
        elif action == "get_nodes":
            nodes = k8s_core_v1.list_node()
            output = "NAME\tSTATUS\tROLES\tAGE\n"
            for node in nodes.items:
                status = "Ready" if any(c.type == "Ready" and c.status == "True" for c in node.status.conditions) else "NotReady"
                roles = ",".join(node.metadata.labels.get("kubernetes.io/role", "worker").split(","))
                age = get_relative_age(node.metadata.creation_timestamp)
                output += f"{node.metadata.name}\t{status}\t{roles}\t{age}\n"
        
        elif action == "get_deployments":
            namespace = action_data.get("params", {}).get("namespace", "all")
            if namespace == "all":
                deployments = k8s_apps_v1.list_deployment_for_all_namespaces()
            else:
                deployments = k8s_apps_v1.list_namespaced_deployment(namespace=namespace)
            
            output = "NAME\tNAMESPACE\tREADY\tUP-TO-DATE\tAVAILABLE\n"
            for dep in deployments.items:
                ready = dep.status.ready_replicas or 0
                replicas = dep.spec.replicas or 0
                available = dep.status.available_replicas or 0
                output += f"{dep.metadata.name}\t{dep.metadata.namespace}\t{ready}/{replicas}\t{replicas}\t{available}\n"
        
        elif action == "get_replicasets":
            namespace = action_data.get("params", {}).get("namespace", "all")
            if namespace == "all":
                replicasets = k8s_apps_v1.list_replica_set_for_all_namespaces()
            else:
                replicasets = k8s_apps_v1.list_namespaced_replica_set(namespace=namespace)
            
            output = "NAME\tNAMESPACE\tDESIRED\tCURRENT\tREADY\tOWNER\n"
            for rs in replicasets.items:
                owner = rs.metadata.owner_references[0].name if rs.metadata.owner_references else "None"
                output += f"{rs.metadata.name}\t{rs.metadata.namespace}\t{rs.spec.replicas}\t{rs.status.replicas or 0}\t{rs.status.ready_replicas or 0}\t{owner}\n"
        
        elif action == "get_events":
            namespace = action_data.get("params", {}).get("namespace", "default")
            events = k8s_core_v1.list_namespaced_event(namespace=namespace)
            
            output = "TYPE\tREASON\tOBJECT\tMESSAGE\tTIME\n"
            for event in events.items[-10:]:  # Last 10 events
                obj_name = f"{event.involved_object.kind}/{event.involved_object.name}"
                output += f"{event.type}\t{event.reason}\t{obj_name}\t{event.message[:50]}...\t{event.last_timestamp}\n"
        
        elif action == "check_pod_dependencies":
            pod_name = action_data.get("params", {}).get("pod")
            namespace = action_data.get("params", {}).get("namespace", "default")
            
            if pod_name:
                # Get pod details
                pod = k8s_core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                
                # Find owning ReplicaSet/Deployment
                output = f"Pod Dependencies for {pod_name}:\n"
                if pod.metadata.owner_references:
                    for owner in pod.metadata.owner_references:
                        output += f"Owned by: {owner.kind}/{owner.name}\n"
                        
                        if owner.kind == "ReplicaSet":
                            rs = k8s_apps_v1.read_namespaced_replica_set(name=owner.name, namespace=namespace)
                            if rs.metadata.owner_references:
                                for rs_owner in rs.metadata.owner_references:
                                    output += f"  Which is owned by: {rs_owner.kind}/{rs_owner.name}\n"
                else:
                    output += "No owner references found\n"
            else:
                output = "Error: Pod name required"
        
        else:
            output = f"Unknown action: {action}"
        
        # Update command history
        command_history = state.get("command_history", [])
        command_history.append({
            "step": state.get("current_step", 1),
            "action": action,
            "params": action_data.get("params", {}),
            "output": output[:500]  # Truncate for history
        })
        
        return {
            "current_output": output,
            "command_history": command_history,
            "current_step": state.get("current_step", 1) + 1
        }
        
    except json.JSONDecodeError:
        return {"current_output": "Error: Invalid action format"}
    except client.ApiException as e:
        return {"current_output": f"Kubernetes API Error: {e.reason}"}
    except Exception as e:
        logger.error(f"Error executing operation: {e}")
        return {"current_output": f"Error: {str(e)}"}



def analysis_agent(state: GraphState):
    """AI agent that analyzes command output and determines next steps."""
    logger.info("Analyzing results")
    
    if not llm or not state.get("next_action_needed", True):
        return {"analysis_result": "Analysis complete", "next_action_needed": False}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an expert Kubernetes analyst. Analyze the command output and determine if more investigation is needed.

Based on the output, provide:
1. Analysis of what the data shows
2. Whether more commands are needed
3. If needed, suggest the next action

Output JSON: {{"analysis": "your analysis", "needs_more": true/false, "next_suggestion": "suggested action"}}'''),
        ("human", "Current output:\n{output}\n\nUser's original request: {user_message}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "output": state["current_output"],
        "user_message": state["user_message"]
    })
    
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response.content)
        if json_match:
            analysis = json.loads(json_match.group())
            return {
                "analysis_result": analysis.get("analysis", ""),
                "next_action_needed": analysis.get("needs_more", False)
            }
    except:
        pass
    
    return {"analysis_result": "Analysis complete", "next_action_needed": False}

def final_formatter_agent(state: GraphState):
    """Formats the final comprehensive response."""
    logger.info("Creating final response")
    
    command_history = state.get("command_history", [])
    
    if not command_history:
        return {"final_response": "No operations performed."}
    
    # Build comprehensive response
    final_response = f"🔍 **Investigation Complete** ({len(command_history)} steps)\n\n"
    
    for i, cmd in enumerate(command_history, 1):
        action_name = cmd['action'].replace('_', ' ').title()
        final_response += f"### Step {i}: {action_name}\n\n"
        
        # Format output as table if possible
        output = cmd['output']
        lines = output.strip().split('\n')
        
        if len(lines) > 1 and '\t' in lines[0]:
            try:
                header = lines[0].split('\t')
                rows = []
                for line in lines[1:5]:  # Limit to first 5 rows
                    if line.strip():
                        row_data = line.split('\t')
                        while len(row_data) < len(header):
                            row_data.append('')
                        rows.append(row_data[:len(header)])
                
                if rows:
                    table = f"| {' | '.join(header)} |\n"
                    table += f"|{'|'.join(['---'] * len(header))}|\n"
                    for row in rows:
                        table += f"| {' | '.join(row)} |\n"
                    final_response += f"{table}\n"
                else:
                    final_response += f"```\n{output[:300]}...\n```\n\n"
            except:
                final_response += f"```\n{output[:300]}...\n```\n\n"
        else:
            final_response += f"```\n{output[:300]}...\n```\n\n"
    
    # Add analysis if available
    if state.get("analysis_result"):
        final_response += f"### 📊 Analysis\n\n{state['analysis_result']}\n\n"
    
    return {"final_response": final_response}

def should_continue(state: GraphState) -> str:
    """Determines if more actions are needed."""
    if state.get("next_action_needed", False) and state.get("current_step", 1) < state.get("max_steps", 5):
        return "continue"
    else:
        return "finish"

# --- LangGraph Workflow Definition ---
workflow = StateGraph(GraphState)
workflow.add_node("context", context_agent)
workflow.add_node("planner", command_planner_agent)
workflow.add_node("executor", action_executor_agent)
workflow.add_node("analyzer", analysis_agent)
workflow.add_node("formatter", final_formatter_agent)

workflow.set_entry_point("context")
workflow.add_edge("context", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "analyzer")
workflow.add_conditional_edges(
    "analyzer",
    should_continue,
    {
        "continue": "planner",
        "finish": "formatter"
    }
)
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
        "current_step": 0,
        "max_steps": 5,
        "command_history": [],
        "current_action": "",
        "current_output": "",
        "analysis_result": "",
        "next_action_needed": True,
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
            "steps_executed": len(result.get("command_history", [])),
            "command_history": result.get("command_history", [])
        }
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
