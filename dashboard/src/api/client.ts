// API client for SENTINEL backend

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface NetworkFlow {
    src_ip: string;
    dst_ip: string;
    src_port: number;
    dst_port: number;
    protocol: string;
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
    duration: number;
    timestamp?: number;
}

export interface ThreatAlert {
    alert_id: string;
    timestamp: string;
    verdict: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
    confidence: number;
    source_ip: string;
    destination_ip: string;
    threat_type?: string;
    kill_chain_stage?: string;
    explanation: Record<string, any>;
    recommended_actions: string[];
}

export interface AttackGraphNode {
    id: string;
    label: string;
    type: 'host' | 'technique';
    compromised?: boolean;
}

export interface AttackGraphEdge {
    source: string;
    target: string;
    label: string;
}

export interface AttackGraph {
    nodes: AttackGraphNode[];
    edges: AttackGraphEdge[];
}

class APIClient {
    async analyzeTraffic(flows: NetworkFlow[]) {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ flows })
        });
        return response.json();
    }

    async simulateAttack(attackType: string = 'apt', numActions: number = 10) {
        const response = await fetch(
            `${API_BASE}/simulate-attack?attack_type=${attackType}&num_actions=${numActions}`,
            { method: 'POST' }
        );
        return response.json();
    }

    async getMITRECoverage() {
        const response = await fetch(`${API_BASE}/mitre-coverage`);
        return response.json();
    }

    async getTechniqueDetails(techniqueId: string) {
        const response = await fetch(`${API_BASE}/mitre/technique/${techniqueId}`);
        return response.json();
    }

    async getAttackGraph(): Promise<AttackGraph> {
        const response = await fetch(`${API_BASE}/attack-graph`);
        return response.json();
    }

    async sendChatMessage(message: string, context?: Record<string, any>) {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, context })
        });
        return response.json();
    }

    async getHealth() {
        const response = await fetch(`${API_BASE}/health`);
        return response.json();
    }
}

export const api = new APIClient();
export default api;
