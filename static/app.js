document.addEventListener('DOMContentLoaded', () => {
  const chatLog = document.getElementById('chat-log');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('user-input');
  const sendBtn = form.querySelector('button');

  // Helper to append a message to the chat log
  function appendMessage(name, content) {
    const div = document.createElement('div');
    div.className = 'chat-message';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'name';
    nameSpan.textContent = name + ':';
    div.appendChild(nameSpan);
    const textNode = document.createTextNode(' ' + content);
    div.appendChild(textNode);
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // Variables to manage delays and timeouts
  let userTimer = null;
  let npcProcessing = false;
  let autoRunTimer = null;

  // Clear the auto run timer if set
  function clearAutoRunTimer() {
    if (autoRunTimer) {
      clearTimeout(autoRunTimer);
      autoRunTimer = null;
    }
  }

  // Display an array of messages one by one with a delay between each. After
  // showing all messages, invoke onComplete.
  function displayMessages(messages, onComplete, index = 0) {
    if (index >= messages.length) {
      npcProcessing = false;
      if (onComplete) onComplete();
      return;
    }
    npcProcessing = true;
    const msg = messages[index];
    appendMessage(msg.name, msg.content);
    setTimeout(() => {
      displayMessages(messages, onComplete, index + 1);
    }, 2000);
  }

  // Start a timer for the user; if they don't respond within 20s, send an
  // empty message to skip their turn
  function startUserTimer() {
    clearUserTimer();
    userTimer = setTimeout(() => {
      sendStep('').then((data) => {
        processServerResponse(data);
      });
    }, 20000);
  }

  function clearUserTimer() {
    if (userTimer) {
      clearTimeout(userTimer);
      userTimer = null;
    }
  }

  // Send a step request to the server. If msg is empty or falsy, skip user turn.
  function sendStep(msg) {
    const body = msg && msg.trim() ? { message: msg.trim() } : {};
    return fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((res) => res.json())
      .catch((err) => {
        console.error('Failed to send step:', err);
        return { messages: [], end: true };
      });
  }

  // Process the server response: show messages, then decide whether to auto
  // request the next step or wait for user input
  function processServerResponse(data) {
    // Cancel any pending automatic NPC run before handling this response
    clearAutoRunTimer();
    const msgs = data.messages || [];
    let nextSpeaker = null;
    const chatMsgs = [];
    msgs.forEach((m) => {
      if (m.name === '전환') {
        const parts = m.content.split('->');
        if (parts.length > 1) {
          nextSpeaker = parts[1].trim();
        }
        // We also display transition messages in the chat
        chatMsgs.push(m);
      } else {
        chatMsgs.push(m);
      }
    });
    displayMessages(chatMsgs, () => {
      // Re-enable input and send button after displaying messages
      sendBtn.disabled = false;
      input.disabled = false;
      if (data.end) {
        // Conversation has ended
        return;
      }
      if (nextSpeaker && nextSpeaker !== '주인공') {
        // It's an NPC turn; schedule auto run immediately
        clearAutoRunTimer();
        autoRunTimer = setTimeout(() => {
          sendStep('').then((newData) => {
            processServerResponse(newData);
          });
        }, 0);
      } else {
        // It's the user's turn; start the skip timer
        startUserTimer();
      }
    });
  }

  // Initialise the chat
  fetch('/init', { method: 'POST' })
    .then((res) => res.json())
    .then((data) => {
      processServerResponse(data);
    })
    .catch((err) => {
      console.error('Failed to init chat:', err);
    });

  // Handle user message submissions
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    clearUserTimer();
    // Cancel any pending auto NPC run since user is taking over
    clearAutoRunTimer();
    // Disable controls while sending; they'll be re-enabled after server response
    sendBtn.disabled = true;
    input.disabled = true;
    input.value = '';
    sendStep(message).then((data) => {
      processServerResponse(data);
    });
  });
});