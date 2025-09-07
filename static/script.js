document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const sendBtn = document.getElementById("send-btn");

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // Display user message
        addMessage(query, "user");
        userInput.value = "";

        // Show typing indicator
        const typingIndicator = addMessage("LANG_VEDA is typing...", "ai", true);
        sendBtn.disabled = true;

        try {
            // Send query to backend
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Remove typing indicator and show AI response
            chatBox.removeChild(typingIndicator);
            addMessage(data.answer, "ai");

        } catch (error) {
            console.error("Error:", error);
            chatBox.removeChild(typingIndicator);
            addMessage("Sorry, something went wrong. Please try again.", "ai");
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    function addMessage(text, sender, isTyping = false) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        
        const p = document.createElement("p");
        p.textContent = text;
        messageElement.appendChild(p);
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to bottom
        return messageElement;
    }
});

