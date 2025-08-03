"""
Example implementation of a multi‑agent chat room using LangGraph and LangChain.

This script defines a simple state machine that orchestrates a group chat
between four non‑player characters (NPCs) and a user. Each NPC has a distinct
persona and two types of memory:

1. **Day memory** – a string containing events from the day. A portion of this
   memory is shared among all agents (e.g. class activities), and another
   portion can be unique to each agent (e.g. a private conversation or a
   personal observation).
2. **Chat memory** – a buffer of previous chat messages that the agent can
   reference when generating new dialogue. Here we use LangChain’s
   ``ConversationBufferMemory`` to persist chat history for each character.

A separate **controller** node decides which agent should speak next based on
the current state. If the controller decides the user should speak and no
input is received within a timeout window, the controller falls back to the
next most appropriate NPC.

This file provides a runnable example that you can adapt to your own
project. To execute it you must install the required libraries with

    pip install langgraph langchain openai

and provide a valid OpenAI API key in your environment (e.g. via the
``OPENAI_API_KEY`` environment variable). Note that running this script
directly from within the ChatGPT environment may not work because the
dependencies are not installed here; copy the file to your local machine and
run it there.

"""

import asyncio
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

# LangChain imports for models, prompts and memory
try:
    # Use the new langchain_community modules where available (LangChain >= 0.2.0)
    from langchain_community.chat_models import ChatOpenAI  # type: ignore
    from langchain_community.memory import ConversationBufferMemory  # type: ignore
except ImportError:
    # Fallback to older langchain modules for backward compatibility
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# LangGraph imports for stateful workflow management
from langgraph.graph import StateGraph, END


@dataclass
class AgentConfig:
    """Configuration for each NPC or the user in the chat room."""

    name: str
    persona: str
    day_memory: str = ""
    chat_memory: ConversationBufferMemory = field(
        default_factory=lambda: ConversationBufferMemory(return_messages=True)
    )


def build_persona_prompt(agent: AgentConfig, messages: List[Any]) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate for an agent given its persona, day memory and
    conversation history. The prompt instructs the LLM to respond as the agent
    would in a group chat.

    Parameters
    ----------
    agent: AgentConfig
        The configuration for the agent whose message we want to generate.
    messages: List[dict]
        A list of message dictionaries representing the conversation so far.

    Returns
    -------
    ChatPromptTemplate
        A template ready to be invoked by a Chat model.
    """
    # System prompt gives high‑level instructions about persona and memory
    system_prompt = (
        "You are {name}, a character in a group chat. "
        "Your persona is described as follows: {persona}. "
        "You will be shown a log of the day's events and the prior chat messages. "
        "Refer to these memories to craft your next message. "
        "Speak naturally and concisely, using first person pronouns. "
        "Only respond with your message; do not narrate your reasoning."
    )
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "user",
                "Day memory:\n{day_memory}\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Conversation so far:\n{conversation}\n\n"
                "Now respond as {name} in this group chat."
            ),
        ]
    )
    return template.partial(name=agent.name, persona=agent.persona)


async def generate_agent_response(
    state: Dict[str, Any], agent: AgentConfig, llm: ChatOpenAI
) -> Dict[str, Any]:
    """
    Generate a single chat message for an agent and update the state. This
    function will be registered as a node in the LangGraph.

    Parameters
    ----------
    state: Dict[str, Any]
        The current state of the conversation, including messages and memory.
    agent: AgentConfig
        The agent whose turn it is to speak.
    llm: ChatOpenAI
        The language model used to generate the agent's response.

    Returns
    -------
    Dict[str, Any]
        Updated state with the new message appended.
    """
    messages: List[dict] = state["messages"]
    # Format chat history for the prompt and update the agent's memory
    # Build a string representation of this agent's chat memory. Depending on
    # the LangChain version, the buffer may contain tuples of (input, output)
    # or individual ChatMessage objects. We extract the content safely.
    chat_lines = []
    for entry in agent.chat_memory.buffer:
        try:
            # Try to unpack a tuple (input, output)
            human, ai = entry  # type: ignore[misc]
            # Append both sides of the exchange if available
            chat_lines.append(getattr(human, "content", str(human)))
            chat_lines.append(getattr(ai, "content", str(ai)))
        except Exception:
            # Fallback: single ChatMessage or unknown format
            if hasattr(entry, "content"):
                chat_lines.append(entry.content)
            else:
                chat_lines.append(str(entry))
    chat_history_str = "\n".join(chat_lines)
    conversation_str = "\n".join(
        [f"{m['name']}: {m['content']}" for m in messages]
    )
    prompt = build_persona_prompt(agent, messages)
    chain = prompt | llm
    # Render the prompt and call the model asynchronously
    response = await chain.ainvoke(
        {
            "day_memory": agent.day_memory,
            "chat_history": chat_history_str,
            "conversation": conversation_str,
        }
    )
    # Extract text from the model's message
    text = response.content.strip()
    # Append the new message to the conversation
    messages.append({"role": "assistant", "content": text, "name": agent.name})
    # Update the agent's chat memory
    agent.chat_memory.save_context(
        {"input": conversation_str}, {"output": text}
    )
    state["messages"] = messages
    return state


async def user_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gather input from the user (player) and append it to the conversation. This
    function uses asyncio to enforce a timeout; if the user does not provide
    input within ``USER_TIMEOUT`` seconds, no message is added and control
    returns to the controller.

    Parameters
    ----------
    state: Dict[str, Any]
        The current state of the conversation.

    Returns
    -------
    Dict[str, Any]
        Updated state with the user's message appended when input is provided.
    """
    USER_TIMEOUT = 20  # seconds
    messages: List[dict] = state["messages"]
    print("\n--- Conversation so far ---")
    for m in messages:
        print(f"{m['name']}: {m['content']}")
    print("-------------------------")
    print("\nIt's your turn to speak (press Enter to skip):")
    try:
        # Use asyncio.to_thread to run blocking input in a non‑blocking way
        user_message = await asyncio.wait_for(
            asyncio.to_thread(input, "You: "), USER_TIMEOUT
        )
        if user_message.strip():
            messages.append({"role": "user", "content": user_message, "name": "주인공"})
            state["messages"] = messages
    except asyncio.TimeoutError:
        # No input within timeout – simply skip user's turn
        print("(No input received; skipping user turn)\n")
    return state


async def control_node(state: Dict[str, Any], controller_llm: ChatOpenAI, agent_names: List[str]) -> Dict[str, Any]:
    """
    Decide which agent should speak next. This controller has access to the
    entire conversation history and all day memories. It prompts an LLM to
    select the next speaker by name. If the LLM returns a name not in the
    provided list, the conversation ends.

    Parameters
    ----------
    state: Dict[str, Any]
        The current state of the conversation.
    controller_llm: ChatOpenAI
        The language model used to decide the next speaker.
    agent_names: List[str]
        A list of all possible speaker names, including the user (주인공).

    Returns
    -------
    Dict[str, Any]
        Updated state with the chosen next speaker stored in ``state['next_speaker']``.
    """
    messages: List[dict] = state["messages"]
    # Decide which names the controller can choose from. By default, allow all
    # participants. However, if the last speaker was an NPC and the message
    # does not contain a question mark or directly reference the user, bias
    # towards other NPCs by removing the user from the candidate list. This
    # encourages NPC‑to‑NPC conversation unless the NPC addresses the user.
    candidate_names = list(agent_names)
    if messages:
        last_msg = messages[-1]
        last_name = last_msg.get("name")
        # Bias away from the user unless directly addressed
        if last_name != "주인공":
            content = str(last_msg.get("content", "")).lower()
            if "주인공" not in content and "?" not in content:
                candidate_names = [n for n in candidate_names if n != "주인공"]
        # Avoid selecting the same speaker twice in a row unless it's the only option
        if last_name in candidate_names and len(candidate_names) > 1:
            candidate_names = [n for n in candidate_names if n != last_name]
    # Build a system prompt instructing the controller to choose the next speaker.
    # We do not mention 'END' here, as the chat will end only via the turn limit.
    system_message = SystemMessage(
        content=(
            "You are the controller for a group chat between several characters. "
            "Given the conversation so far, choose which character should speak next. "
            "Respond only with the character's name from the following list: {names}."
        )
    )
    # Format the conversation into ChatMessages for the LLM
    chat_messages = [system_message]
    for m in messages:
        if m["role"] == "user":
            chat_messages.append(HumanMessage(content=m["content"]))
        else:
            # Represent all NPC replies as AI messages
            chat_messages.append(AIMessage(content=m["content"]))
    # Insert the candidate names into the system prompt
    formatted_system = SystemMessage(
        content=system_message.content.format(names=", ".join(candidate_names))
    )
    chat_messages[0] = formatted_system
    # Call the LLM synchronously for control decisions
    response = await controller_llm.ainvoke(chat_messages)
    next_speaker = response.content.strip()
    # If the model picks an invalid name (outside the allowed names), end the chat
    if next_speaker not in candidate_names and next_speaker != "END":
        next_speaker = "END"
    state["next_speaker"] = next_speaker
    return state


def create_graph(agents: Dict[str, AgentConfig], llm: ChatOpenAI, controller_llm: ChatOpenAI) -> StateGraph:
    """
    Build the LangGraph for the multi‑agent chat scenario. Each NPC agent is
    represented by a node, and there is a user input node and a controller
    node. The controller decides the next speaker and directs the flow of the
    conversation.

    Parameters
    ----------
    agents: Dict[str, AgentConfig]
        Mapping of agent names to their configurations.
    llm: ChatOpenAI
        The language model used by NPCs to generate responses.
    controller_llm: ChatOpenAI
        The language model used by the controller to select the next speaker.

    Returns
    -------
    StateGraph
        A compiled graph that can be invoked with an initial state.
    """
    workflow = StateGraph(dict)
    # Add controller node
    workflow.add_node(
        "controller", lambda state: control_node(state, controller_llm, list(agents.keys()))
    )
    # Add NPC nodes
    for name, agent_config in agents.items():
        if name == "주인공":
            # User node handles input separately
            workflow.add_node(name, user_input_node)
        else:
            # Bind the agent config and model to the generation function
            workflow.add_node(
                name,
                partial(generate_agent_response, agent=agent_config, llm=llm),
            )
    # Edges: after any agent speaks, go back to controller
    for name in agents.keys():
        workflow.add_edge(name, "controller")
    # Conditional routing from controller to the next speaker
    # The controller writes 'next_speaker' into state; use it to choose a node
    choices = {name: name for name in agents.keys()}
    choices["END"] = END
    workflow.add_conditional_edges(
        "controller",
        lambda state: state.get("next_speaker", "END"),
        choices,
    )
    # Set entry point
    workflow.set_entry_point("controller")
    return workflow


async def run_chat():
    """Entry point to run the chat graph with example day memories and personas.

    When the session finishes, all agent memories and the full conversation log
    are saved into a new subfolder under ``chat_logs``. The folder name is
    derived from the current timestamp (e.g. ``chat_logs/session_20250101_120000``).
    """
    # Initialize language models for NPCs and controller
    npc_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    controller_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    # Define agent configurations
    agents: Dict[str, AgentConfig] = {
        "금상재": AgentConfig(
            name="금상재",
            persona="양아치처럼 보이는 거칠고 무뚝뚝한 타입. 은근하게 배려심 있음.",
            day_memory="오늘 수업 후 텃밭에서 모시현과 같이 상추를 심었다. 너는 그 모습을 멀리서 지켜봤다.",
        ),
        "강시아": AgentConfig(
            name="강시아",
            persona="말 수 적고 감성적. 창밖을 자주 봄. 사색적이며 시적인 표현 자주 사용.",
            day_memory="쉬는 시간에 혼자 창밖을 보며 비 내리는 풍경을 스케치북에 그렸다.",
        ),
        "모시현": AgentConfig(
            name="모시현",
            persona="안경 쓴 모범생. 규칙적이고 사리분별 명확. 말투 단정.",
            day_memory="텃밭에서 금상재에게 상추 심는 방법을 알려주며 질서를 유지하려 했다.",
        ),
        "하인호": AgentConfig(
            name="하인호",
            persona="친화력 좋고 유쾌한 분위기 메이커. 중립적인 입장 잘 유지.",
            day_memory="하교길에 모두에게 과자를 나눠주며 웃음꽃을 피웠다.",
        ),
        "주인공": AgentConfig(
            name="주인공",
            persona="전학생이자 반장. 소심하지만 책임감 있음. 반장으로서 조심스럽게 중심을 잡으려 함.",
            day_memory="첫 날이라 긴장했지만, 친구들이 친절하게 대해줘 조금 안심했다.",
        ),
    }
    # Construct the graph
    workflow = create_graph(agents, npc_llm, controller_llm)
    graph = workflow.compile()
    # Initial state with empty messages and next speaker unset
    state = {"messages": [], "next_speaker": None}
    # Run the chat until an END state is reached
    await graph.ainvoke(state)

    # After the chat ends, save memories and conversation log
    save_session_logs(agents, state)


def save_session_logs(agents: Dict[str, AgentConfig], state: Dict[str, Any]) -> None:
    """
    Persist the agent memories and the full conversation transcript to disk.

    This helper is invoked at the end of a chat session. It creates a folder
    ``chat_logs/session_<timestamp>`` in the current working directory. Inside
    the folder it writes:

    * ``conversation.txt`` – the full chronological log of the conversation.
    * ``<agent>_day_memory.txt`` – the day memory string for each agent.
    * ``<agent>_chat_memory.txt`` – the chat memory buffer for each agent.

    Parameters
    ----------
    agents: Dict[str, AgentConfig]
        All agents involved in the conversation.
    state: Dict[str, Any]
        The final state containing the full conversation under ``messages``.
    """
    import datetime
    import pathlib
    # Build the base directory and session folder
    base_dir = pathlib.Path("chat_logs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    # Save the full conversation transcript
    conv_file = session_dir / "conversation.txt"
    with conv_file.open("w", encoding="utf-8") as f:
        for msg in state.get("messages", []):
            f.write(f"{msg['name']}: {msg['content']}\n")
    # Save individual agent memories
    for name, agent in agents.items():
        # Day memory
        day_path = session_dir / f"{name}_day_memory.txt"
        with day_path.open("w", encoding="utf-8") as f:
            f.write(agent.day_memory)
        # Chat memory
        chat_path = session_dir / f"{name}_chat_memory.txt"
        with chat_path.open("w", encoding="utf-8") as f:
            # Each memory entry is stored as a tuple of (input, output) in the buffer
            for human, ai in agent.chat_memory.buffer:
                f.write(f"User: {human.content}\n")
                f.write(f"Agent {name}: {ai.content}\n\n")
    print(f"Session logs saved to {session_dir}")


if __name__ == "__main__":
    # Only run the chat loop when executed directly. When imported, the
    # functions and classes defined above can be used programmatically.
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nChat terminated by user.")