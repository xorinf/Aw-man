import { useState, useEffect, useCallback, useRef } from 'react';
import type { ThreatAlert } from '../api/client';

interface WebSocketMessage {
    type: 'connection' | 'alert' | 'heartbeat' | 'pong';
    message?: string;
    data?: ThreatAlert;
    timestamp: string;
}

export function useWebSocket(url: string) {
    const [isConnected, setIsConnected] = useState(false);
    const [alerts, setAlerts] = useState<ThreatAlert[]>([]);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number>();

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                console.log('[WebSocket] Connected to SENTINEL');
                setIsConnected(true);
            };

            ws.onmessage = (event) => {
                const message: WebSocketMessage = JSON.parse(event.data);
                setLastMessage(message);

                if (message.type === 'alert' && message.data) {
                    setAlerts((prev) => [message.data!, ...prev].slice(0, 100)); // Keep last 100 alerts

                    // Show browser notification if permitted
                    if (Notification.permission === 'granted') {
                        new Notification('🚨 SENTINEL Alert', {
                            body: `${message.data.verdict}: ${message.data.threat_type || 'Unknown threat'}`,
                            icon: '/sentinel-icon.svg',
                            badge: '/sentinel-icon.svg'
                        });
                    }
                }
            };

            ws.onerror = (error) => {
                console.error('[WebSocket] Error:', error);
            };

            ws.onclose = () => {
                console.log('[WebSocket] Disconnected');
                setIsConnected(false);

                // Attempt reconnect after 3 seconds
                reconnectTimeoutRef.current = window.setTimeout(() => {
                    console.log('[WebSocket] Attempting to reconnect...');
                    connect();
                }, 3000);
            };

            wsRef.current = ws;
        } catch (error) {
            console.error('[WebSocket] Connection failed:', error);
            reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
        }
    }, [url]);

    useEffect(() => {
        connect();

        // Request notification permission
        if (Notification.permission === 'default') {
            Notification.requestPermission();
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
        };
    }, [connect]);

    const sendMessage = useCallback((message: string) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(message);
        }
    }, []);

    return {
        isConnected,
        alerts,
        lastMessage,
        sendMessage
    };
}
