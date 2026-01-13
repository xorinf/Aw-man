import React, { useState } from 'react';
import { Send, Sparkles } from 'lucide-react';
import api from '../api/client';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    suggestions?: string[];
}

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'assistant',
            content: "Hello! I'm your AI Security Copilot. I can help you analyze threats, explain detection decisions, and recommend response actions. What would you like to investigate?",
            timestamp: new Date(),
            suggestions: ['Show latest alerts', 'Explain last detection', 'Run attack simulation']
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async (text: string) => {
        if (!text.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: text,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await api.sendChatMessage(text);

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: response.response,
                timestamp: new Date(),
                suggestions: response.suggestions
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSuggestionClick = (suggestion: string) => {
        sendMessage(suggestion);
    };

    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.role}`}>
                        <div className="message-content">
                            {msg.role === 'assistant' && (
                                <Sparkles size={16} className="assistant-icon" />
                            )}
                            <div className="message-text">{msg.content}</div>
                        </div>
                        {msg.suggestions && (
                            <div className="suggestions">
                                {msg.suggestions.map((suggestion, i) => (
                                    <button
                                        key={i}
                                        className="btn btn-secondary suggestion-btn"
                                        onClick={() => handleSuggestionClick(suggestion)}
                                    >
                                        {suggestion}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div className="message assistant">
                        <div className="message-content">
                            <Sparkles size={16} className="assistant-icon" />
                            <div className="spinner" />
                        </div>
                    </div>
                )}
            </div>

            <div className="chat-input-container">
                <input
                    type="text"
                    className="input chat-input"
                    placeholder="Ask about threats, alerts, or security recommendations..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage(input)}
                    disabled={isLoading}
                />
                <button
                    className="btn btn-primary send-btn"
                    onClick={() => sendMessage(input)}
                    disabled={isLoading || !input.trim()}
                >
                    <Send size={18} />
                </button>
            </div>

            <style>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          max-height: 600px;
        }

        .messages {
          flex: 1;
          overflow-y: auto;
          padding: var(--spacing-lg);
          display: flex;
          flex-direction: column;
          gap: var(--spacing-lg);
        }

        .message {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
          max-width: 80%;
        }

        .message.user {
          align-self: flex-end;
          align-items: flex-end;
        }

        .message.assistant {
          align-self: flex-start;
        }

        .message-content {
          display: flex;
          gap: var(--spacing-sm);
          align-items: flex-start;
        }

        .assistant-icon {
          color: var(--color-primary);
          flex-shrink: 0;
          margin-top: 4px;
        }

        .message-text {
          background: var(--color-surface);
          padding: var(--spacing-md);
          border-radius: var(--radius-md);
          white-space: pre-wrap;
          line-height: 1.6;
        }

        .message.user .message-text {
          background: linear-gradient(135deg, var(--color-primary), #6d28d9);
        }

        .suggestions {
          display: flex;
          flex-wrap: wrap;
          gap: var(--spacing-sm);
          margin-top: var(--spacing-sm);
        }

        .suggestion-btn {
          font-size: var(--font-size-xs);
          padding: 4px 12px;
        }

        .chat-input-container {
          display: flex;
          gap: var(--spacing-sm);
          padding: var(--spacing-lg);
          border-top: 1px solid rgba(124, 58, 237, 0.2);
          background: var(--color-bg-secondary);
        }

        .chat-input {
          flex: 1;
        }

        .send-btn {
          flex-shrink: 0;
        }
      `}</style>
        </div>
    );
}
