// static/js/script.js (UPDATED)

document.addEventListener('DOMContentLoaded', () => {
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const currentDateSpan = document.getElementById('current-date');
    const operatorIdInput = document.getElementById('operator-id-input'); // Get the new input

    // Display current date in the welcome message
    const today = new Date();
    currentDateSpan.textContent = today.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // Function to append messages to the chat display
    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (sender === 'You') {
            messageDiv.classList.add('user-message');
        } else {
            messageDiv.classList.add('bot-message');
        }
        messageDiv.textContent = message;
        chatDisplay.appendChild(messageDiv);
        chatDisplay.scrollTop = chatDisplay.scrollHeight; // Auto-scroll to bottom
    }

    // Function to send message to Flask backend
    async function sendMessage() {
        const message = userInput.value.trim();
        const operatorId = parseInt(operatorIdInput.value, 10); // Get operator ID as integer

        if (!message) return;

        if (isNaN(operatorId) || operatorId <= 0) {
            alert("Please enter a valid Operator ID (a positive number).");
            operatorIdInput.focus();
            return;
        }

        appendMessage('You', message);
        userInput.value = ''; // Clear input

        // Disable input and button while processing
        userInput.disabled = true;
        sendButton.disabled = true;
        operatorIdInput.disabled = true; // Also disable operator ID input
        appendMessage('RedBusBot', 'Thinking...'); // Show thinking indicator

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    operator_id: operatorId // Send the operator ID
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // Remove the 'Thinking...' message before appending the actual response
            const thinkingMessage = chatDisplay.querySelector('.bot-message:last-child');
            if (thinkingMessage && thinkingMessage.textContent === 'Thinking...') {
                chatDisplay.removeChild(thinkingMessage);
            }
            appendMessage('RedBusBot', data.response);

        } catch (error) {
            console.error('Error sending message:', error);
            const thinkingMessage = chatDisplay.querySelector('.bot-message:last-child');
            if (thinkingMessage && thinkingMessage.textContent === 'Thinking...') {
                chatDisplay.removeChild(thinkingMessage);
            }
            appendMessage('RedBusBot', 'Oops! Something went wrong. Please try again later.');
            alert('Error communicating with the chatbot. See console for details.');
        } finally {
            // Re-enable input and button
            userInput.disabled = false;
            sendButton.disabled = false;
            operatorIdInput.disabled = false; // Re-enable operator ID input
            userInput.focus(); // Set focus back on input field
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Set initial focus
    userInput.focus();
});