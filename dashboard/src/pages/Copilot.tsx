import React from 'react';
import { ChatInterface } from '../components/ChatInterface';
import { useAlerts } from '../context/AlertContext';

export function Copilot() {
    const { alerts } = useAlerts();
    const recentAlerts = alerts.slice(0, 3);

    return (
        <div className="page-container copilot-page">
            <div className="page-header">
                <h1 className="page-title">AI Security Copilot</h1>
                <p className="page-subtitle">Ask questions, get insights, and receive security recommendations</p>
            </div>

            <div className="copilot-layout">
                <div className="chat-panel card">
                    <ChatInterface />
                </div>

                <div className="context-panel">
                    <div className="card context-section">
                        <h3>Recent Activity</h3>
                        <div className="activity-list">
                            {recentAlerts.map(alert => (
                                <div key={alert.alert_id} className="activity-item">
                                    <span className={`badge badge-${alert.verdict.toLowerCase()}`}>
                                        {alert.verdict}
                                    </span>
                                    <div className="activity-details">
                                        <div className="activity-title">{alert.threat_type || 'Unknown Threat'}</div>
                                        <div className="activity-meta">{alert.source_ip} → {alert.destination_ip}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card context-section">
                        <h3>Quick Actions</h3>
                        <div className="quick-actions">
                            <button className="btn btn-secondary action-btn">Run Simulation</button>
                            <button className="btn btn-secondary action-btn">View Attack Graph</button>
                            <button className="btn btn-secondary action-btn">MITRE Coverage</button>
                            <button className="btn btn-secondary action-btn">Generate Report</button>
                        </div>
                    </div>
                </div>
            </div>

            <style>{`
        .copilot-page {
          height: calc(100vh - 4rem);
        }

        .copilot-layout {
          display: grid;
          grid-template-columns: 1fr 350px;
          gap: var(--spacing-xl);
          height: calc(100% - 120px);
        }

        .chat-panel {
          display: flex;
          flex-direction: column;
          height: 100%;
        }

        .context-panel {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-lg);
          overflow-y: auto;
        }

        .context-section h3 {
          font-size: var(--font-size-lg);
          margin-bottom: var(--spacing-md);
          color: var(--color-text-secondary);
        }

        .activity-list {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        .activity-item {
          display: flex;
          gap: var(--spacing-sm);
          padding: var(--spacing-sm);
          background: var(--color-bg-secondary);
          border-radius: var(--radius-md);
        }

        .activity-details {
          flex: 1;
        }

        .activity-title {
          font-size: var(--font-size-sm);
          font-weight: 500;
          margin-bottom: 2px;
        }

        .activity-meta {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
          font-family: 'Courier New', monospace;
        }

        .quick-actions {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        .action-btn {
          width: 100%;
          justify-content: flex-start;
        }

        @media (max-width: 1024px) {
          .copilot-layout {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
        </div>
    );
}
