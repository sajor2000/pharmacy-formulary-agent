/**
 * Pharmacy Formulary Helper
 * Main JavaScript file for the web interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown renderer
    const md = window.markdownit({
        html: true,
        linkify: true,
        typographer: true
    });

    // Elements
    const chatContainer = document.getElementById('chatContainer');
    const queryInput = document.getElementById('queryInput');
    const sendButton = document.getElementById('sendButton');
    const insuranceBadges = document.querySelectorAll('.insurance-badge');
    const medicationClassBadges = document.querySelectorAll('.medication-class-badge');
    const clearButton = document.getElementById('clearChat');
    
    // Selected filters
    let selectedInsurance = null;
    let selectedMedicationClass = null;
    
    // Handle insurance selection
    insuranceBadges.forEach(badge => {
        badge.addEventListener('click', function() {
            // Remove active class from all badges
            insuranceBadges.forEach(b => b.classList.remove('active'));
            
            // If clicking the same badge, deselect it
            if (selectedInsurance === this.dataset.insurance) {
                selectedInsurance = null;
            } else {
                // Otherwise, select the new badge
                this.classList.add('active');
                selectedInsurance = this.dataset.insurance;
            }
            
            updateFilterDisplay();
        });
    });
    
    // Handle medication class selection
    medicationClassBadges.forEach(badge => {
        badge.addEventListener('click', function() {
            // Remove active class from all badges
            medicationClassBadges.forEach(b => b.classList.remove('active'));
            
            // If clicking the same badge, deselect it
            if (selectedMedicationClass === this.dataset.class) {
                selectedMedicationClass = null;
            } else {
                // Otherwise, select the new badge
                this.classList.add('active');
                selectedMedicationClass = this.dataset.class;
            }
            
            updateFilterDisplay();
        });
    });
    
    // Update filter display
    function updateFilterDisplay() {
        const filterDisplay = document.getElementById('filterDisplay');
        let filterText = '';
        
        if (selectedInsurance) {
            filterText += `Insurance: ${selectedInsurance} `;
        }
        
        if (selectedMedicationClass) {
            filterText += `Class: ${selectedMedicationClass}`;
        }
        
        filterDisplay.textContent = filterText;
    }
    
    // Handle send button click
    sendButton.addEventListener('click', function() {
        submitQuery();
    });
    
    // Handle enter key press
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            submitQuery();
        }
    });
    
    // Clear chat
    clearButton.addEventListener('click', function() {
        // Clear chat except for the welcome message
        const welcomeMessage = chatContainer.firstElementChild;
        chatContainer.innerHTML = '';
        chatContainer.appendChild(welcomeMessage);
    });
    
    // Submit query to backend
    function submitQuery() {
        const query = queryInput.value.trim();
        
        if (!query) {
            return;
        }
        
        // Add user message to chat
        addUserMessage(query);
        
        // Clear input
        queryInput.value = '';
        
        // Show loading indicator
        addLoadingMessage();
        
        // Send query to backend
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: query,
                insurance: selectedInsurance,
                medication_class: selectedMedicationClass
            })
        })
        .then(response => {
            // Check if response is ok before trying to parse JSON
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Remove loading indicator
            removeLoadingMessage();
            
            if (data.error) {
                addSystemMessage(`Sorry, there was an error processing your request: ${data.error}`);
            } else {
                addAssistantMessage(data.response);
            }
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            // Remove loading indicator
            removeLoadingMessage();
            
            console.error('Error during fetch operation:', error);
            addSystemMessage(`Sorry, there was an error processing your request: ${error.message}`);
            
            // Scroll to bottom
            scrollToBottom();
        });
    }
    
    // Add user message to chat
    function addUserMessage(message) {
        const messageRow = document.createElement('div');
        messageRow.className = 'message-row user-row';
        
        messageRow.innerHTML = `
            <div class="message user-message">
                <div class="message-content">
                    <p>${escapeHtml(message)}</p>
                </div>
            </div>
            <div class="avatar user-avatar">
                <i class="bi bi-person"></i>
            </div>
        `;
        
        chatContainer.appendChild(messageRow);
        scrollToBottom();
    }
    
    // Add assistant message to chat
    function addAssistantMessage(message) {
        const messageRow = document.createElement('div');
        messageRow.className = 'message-row assistant-row';
        
        // Render markdown
        const renderedContent = md.render(message);
        
        messageRow.innerHTML = `
            <div class="avatar assistant-avatar">
                <i class="bi bi-robot"></i>
            </div>
            <div class="message assistant-message">
                <div class="message-content markdown-content">
                    ${renderedContent}
                </div>
            </div>
        `;
        
        chatContainer.appendChild(messageRow);
        scrollToBottom();
    }
    
    // Add system message to chat
    function addSystemMessage(message) {
        const messageRow = document.createElement('div');
        messageRow.className = 'message-row system-row';
        
        messageRow.innerHTML = `
            <div class="system-message">
                ${escapeHtml(message)}
            </div>
        `;
        
        chatContainer.appendChild(messageRow);
        scrollToBottom();
    }
    
    // Add loading message
    function addLoadingMessage() {
        const loadingRow = document.createElement('div');
        loadingRow.className = 'message-row assistant-row loading-row';
        loadingRow.id = 'loadingMessage';
        
        loadingRow.innerHTML = `
            <div class="avatar assistant-avatar">
                <i class="bi bi-robot"></i>
            </div>
            <div class="message assistant-message">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        chatContainer.appendChild(loadingRow);
        scrollToBottom();
    }
    
    // Remove loading message
    function removeLoadingMessage() {
        const loadingMessage = document.getElementById('loadingMessage');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
