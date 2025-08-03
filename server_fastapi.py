"""
FastAPI backend for the 15일 반장 채팅 데모 using a step‑based dialogue flow.

This server orchestrates a group chat between several NPC agents and the user
by advancing exactly one step of the conversation per request. Each step
consists of an agent (NPC or user) generating a single message followed by a
controller decision that selects the next speaker. The server returns both
the agent's message and a transition string ("전환: A -> B") to the client,
which can then decide whether to immediately request another step (for NPC
turns) or wait for user input.

Endpoints:

* ``GET /`` – serve the chat UI (index.html).
* ``POST /init`` – reset the conversation and produce the first NPC message and
  transition. Returns a list of messages and an ``end`` flag.
* ``POST /step`` – advance one step in the conversation. Accepts an optional
  ``message`` field (the user's message). Returns the agent message and
  transition for this step along with an ``end`` flag indicating if the
  conversation has reached its 20‑turn limit.

Static assets (CSS and JavaScript) are served under ``/static`` using
FastAPI's ``StaticFiles``.

To run this server, install the dependencies:

    pip install fastapi uvicorn langgraph langchain openai python-dotenv

Then start the server with:

    uvicorn server_fastapi:app --reload

Navigate to ``http://localhost:8000`` to use the chat interface.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys

# Ensure we can import multi_agent_chat_graph from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from multi_agent_chat_graph import (
    AgentConfig,
    generate_agent_response,
    control_node,
)

try:
    # Prefer the community package for ChatOpenAI
    from langchain_community.chat_models import ChatOpenAI  # type: ignore
except ImportError:
    from langchain.chat_models import ChatOpenAI  # type: ignore

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


app = FastAPI()

# Locate static and template directories relative to this file
base_dir = Path(__file__).resolve().parent
static_dir = base_dir / "static"
template_dir = base_dir / "templates"
if not static_dir.exists():
    alt = base_dir / "web_demo" / "static"
    if alt.exists():
        static_dir = alt
if not template_dir.exists():
    alt = base_dir / "web_demo" / "templates"
    if alt.exists():
        template_dir = alt

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state: agents, conversation state and language models
agents: Dict[str, AgentConfig] = {}
state: Dict[str, Any] = {"messages": [], "next_speaker": None}
npc_llm: ChatOpenAI = None
controller_llm: ChatOpenAI = None

# Session logging
from typing import Optional as Opt
session_dir: Opt[Path] = None


def init_agents() -> None:
    """Initialise agent configurations and language models."""
    global agents, npc_llm, controller_llm
    api_key = os.getenv("OPENAI_API_KEY")
    if npc_llm is None:
        npc_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
    if controller_llm is None:
        controller_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
    agents = {
        "금상재": AgentConfig(
            name="금상재",
            persona=(
                "양아치처럼 보이고 거칠며 무뚝뚝한 성격이다. 하지만 속정이 깊어 친구들을 은근히 챙긴다. "
                "쓰레기 같은 농담과 허세 섞인 말투로 분위기를 띄우지만, 마음에 드는 사람에겐 진심으로 응원한다."
            ),
            day_memory="오늘 수업 후 텃밭에서 모시현과 같이 상추를 심었다. 너는 그 모습을 멀리서 지켜봤다.",
        ),
        "강시아": AgentConfig(
            name="강시아",
            persona=(
                "말 수는 적지만 감성적이고 시적인 표현을 즐긴다. 창밖을 바라보며 자연을 관찰하는 것을 좋아하고, "
                "때때로 깊은 생각에 잠긴다. 사소한 것에서 의미를 찾는 섬세한 면이 있다."
            ),
            day_memory="쉬는 시간에 혼자 창밖을 보며 비 내리는 풍경을 스케치북에 그렸다.",
        ),
        "모시현": AgentConfig(
            name="모시현",
            persona=(
                "안경을 쓰고 책임감이 강한 모범생이다. 규칙과 질서를 중요하게 생각하지만 친구들에게는 따뜻한 조언을 아끼지 않는다. "
                "궁금한 것이 있으면 꼬치꼬치 캐묻는 편이다."
            ),
            day_memory="텃밭에서 금상재에게 상추 심는 방법을 알려주며 질서를 유지하려 했다.",
        ),
        "하인호": AgentConfig(
            name="하인호",
            persona=(
                "친화력이 좋고 유머 감각이 뛰어난 분위기 메이커다. 누구와도 스스럼없이 대화하며 갈등을 중재하고, "
                "재치 있는 말로 분위기를 살린다."
            ),
            day_memory="하교길에 모두에게 과자를 나눠주며 웃음꽃을 피웠다.",
        ),
        "주인공": AgentConfig(
            name="주인공",
            persona=(
                "전학생이자 반장. 소심하지만 책임감이 강하고 친구들과 가까워지기 위해 노력한다. "
                "때로는 부끄러워하지만 솔직하게 의견을 말하려 하며, 모두가 즐겁게 지낼 수 있도록 중심을 잡으려 한다."
            ),
            day_memory="첫 날이라 긴장했지만, 친구들이 친절하게 대해줘 조금 안심했다.",
        ),
    }


def write_session_logs() -> None:
    """Persist the conversation and agent memories to a session folder."""
    global session_dir
    import datetime
    import pathlib
    # Create session folder once
    if session_dir is None:
        base = pathlib.Path("chat_logs")
        base.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir_local = base / f"session_{timestamp}"
        session_dir_local.mkdir(parents=True, exist_ok=True)
        session_dir = session_dir_local
    # Save conversation
    conv_file = session_dir / "conversation.txt"
    with conv_file.open("w", encoding="utf-8") as f:
        for msg in state["messages"]:
            f.write(f"{msg['name']}: {msg['content']}\n")
    # Save memories
    for name, agent in agents.items():
        day_path = session_dir / f"{name}_day_memory.txt"
        with day_path.open("w", encoding="utf-8") as f:
            f.write(agent.day_memory)
        chat_path = session_dir / f"{name}_chat_memory.txt"
        with chat_path.open("w", encoding="utf-8") as f:
            for entry in agent.chat_memory.buffer:
                try:
                    human, ai = entry  # type: ignore[misc]
                    f.write(f"User: {getattr(human, 'content', str(human))}\n")
                    f.write(f"Agent {name}: {getattr(ai, 'content', str(ai))}\n\n")
                except Exception:
                    if hasattr(entry, 'content'):
                        f.write(f"{getattr(entry, 'type', 'msg')}: {entry.content}\n\n")


async def step(user_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Advance the conversation by one step.

    If ``user_message`` is provided, it is appended to the conversation as the
    user's message. The current speaker is determined by ``state['next_speaker']``.
    If there is no current speaker (initialisation), the first NPC is chosen.
    Depending on the current speaker, this function generates an NPC message or
    skips the user turn, then uses the controller to choose the next speaker.

    Returns a dictionary with keys:
        "messages": List of messages generated this step (agent message and
                    transition). Each item is a dict with "name" and "content".
        "end": Bool indicating whether the conversation has reached the end.
    """
    global state
    messages_out: List[Dict[str, str]] = []
    # If conversation has already ended, return nothing
    if state.get("next_speaker") == "END":
        return {"messages": [], "end": True}
    # If a user message is provided and non‑empty, treat this as the user's turn
    # regardless of the current speaker. Append the message and immediately
    # hand control to the controller to pick the next speaker.
    if user_message is not None and user_message.strip():
        state["messages"].append({"role": "user", "content": user_message, "name": "주인공"})
        # Decide next speaker
        state = await control_node(state, controller_llm, list(agents.keys()))
        next_speaker = state.get("next_speaker")
        messages_out.append({"name": "주인공", "content": user_message})
        messages_out.append({"name": "전환", "content": f"주인공 -> {next_speaker}"})
        write_session_logs()
        return {"messages": messages_out, "end": next_speaker == "END"}

    current_speaker = state.get("next_speaker")
    # If no current speaker, choose the first NPC to start
    if current_speaker is None:
        for name in agents:
            if name != "주인공":
                current_speaker = name
                break
        state["next_speaker"] = current_speaker
    # If current speaker is user but no user message provided, skip user turn
    if current_speaker == "주인공":
        # Controller selects next speaker
        state = await control_node(state, controller_llm, list(agents.keys()))
        next_speaker = state.get("next_speaker")
        messages_out.append({"name": "전환", "content": f"주인공 -> {next_speaker}"})
        write_session_logs()
        return {"messages": messages_out, "end": next_speaker == "END"}
    # Otherwise, current speaker is an NPC
    if current_speaker and current_speaker != "주인공":
        # Generate NPC message
        state = await generate_agent_response(state, agents[current_speaker], npc_llm)
        npc_msg = state["messages"][-1]
        # If we've reached 20 messages, end the conversation
        if len(state["messages"]) >= 20:
            state["next_speaker"] = "END"
            messages_out.append({"name": npc_msg["name"], "content": npc_msg["content"]})
            messages_out.append({"name": "전환", "content": f"{current_speaker} -> END"})
            write_session_logs()
            return {"messages": messages_out, "end": True}
        # Otherwise, controller decides next speaker
        state = await control_node(state, controller_llm, list(agents.keys()))
        next_speaker = state.get("next_speaker")
        messages_out.append({"name": npc_msg["name"], "content": npc_msg["content"]})
        messages_out.append({"name": "전환", "content": f"{current_speaker} -> {next_speaker}"})
        write_session_logs()
        return {"messages": messages_out, "end": next_speaker == "END"}
    # Fallback
    return {"messages": [], "end": False}


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    html_path = template_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/init")
async def init_chat() -> JSONResponse:
    """
    Reset the conversation state and return the first NPC message and transition.
    """
    global state, session_dir
    init_agents()
    state = {"messages": [], "next_speaker": None}
    session_dir = None
    result = await step(None)
    return JSONResponse(result)


@app.post("/step")
async def step_endpoint(data: Dict[str, Any]) -> JSONResponse:
    """
    Advance the conversation by one step. The request body may include a
    ``message`` field containing the user's message. If omitted or empty, the
    user turn is skipped.
    """
    user_msg = data.get("message")
    if isinstance(user_msg, str):
        user_msg = user_msg.strip()
    else:
        user_msg = None
    result = await step(user_msg)
    return JSONResponse(result)