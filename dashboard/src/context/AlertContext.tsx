import React, { createContext, useContext, ReactNode } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import type { ThreatAlert } from '../api/client';

interface AlertContextType {
    alerts: ThreatAlert[];
    isConnected: boolean;
    lastMessage: any;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export function AlertProvider({ children }: { children: ReactNode }) {
    const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/alerts';
    const { alerts, isConnected, lastMessage } = useWebSocket(WS_URL);

    return (
        <AlertContext.Provider value={{ alerts, isConnected, lastMessage }}>
            {children}
        </AlertContext.Provider>
    );
}

export function useAlerts() {
    const context = useContext(AlertContext);
    if (!context) {
        throw new Error('useAlerts must be used within AlertProvider');
    }
    return context;
}
