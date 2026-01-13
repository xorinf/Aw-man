import React, { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { ChevronDown, ChevronUp, Shield, Ban, Eye } from 'lucide-react';
import type { ThreatAlert } from '../api/client';

interface AlertCardProps {
    alert: ThreatAlert;
}

export function AlertCard({ alert }: AlertCardProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    const getSeverityColor = (verdict: string) => {
        switch (verdict) {
            case 'CRITICAL':
                return 'critical';
            case 'HIGH':
                return 'high';
            case 'MEDIUM':
                return 'medium';
            case 'LOW':
                return 'low';
            default:
                return 'medium';
        }
    };

    const timeAgo = formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true });

    return (
        <div className={`alert-card ${getSeverityColor(alert.verdict)}`}>
            <div className="alert-header" onClick={() => setIsExpanded(!isExpanded)}>
                <div className="alert-header-left">
                    <span className={`badge badge-${getSeverityColor(alert.verdict)}`}>
                        <Shield size={12} />
                        {alert.verdict}
                    </span>
                    <span className="alert-id">{alert.alert_id}</span>
                </div>
                <div className="alert-header-right">
                    <span className="alert-time">{timeAgo}</span>
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
            </div>

            <div className="alert-body">
                <div className="alert-info">
                    <div className="alert-ips">
                        <span className="ip-label">Source:</span>
                        <code className="ip-address">{alert.source_ip}</code>
                        <span className="separator">→</span>
                        <span className="ip-label">Destination:</span>
                        <code className="ip-address">{alert.destination_ip}</code>
                    </div>
                    {alert.threat_type && (
                        <div className="threat-type">{alert.threat_type}</div>
                    )}
                    {alert.kill_chain_stage && (
                        <div className="kill-chain">
                            Kill Chain: <strong>{alert.kill_chain_stage}</strong>
                        </div>
                    )}
                </div>

                {isExpanded && (
                    <div className="alert-expanded fade-in">
                        <div className="explanation">
                            <h4>Explanation</h4>
                            <div className="confidence">
                                Confidence: <strong>{(alert.confidence * 100).toFixed(1)}%</strong>
                            </div>
                            {alert.explanation.top_factors && (
                                <div className="factors">
                                    {alert.explanation.top_factors.map((factor: any, i: number) => (
                                        <div key={i} className="factor">
                                            <span className="factor-feature">{factor.feature}</span>
                                            <span className="factor-impact">
                                                Impact: {(factor.impact * 100).toFixed(0)}%
                                            </span>
                                            <p className="factor-reason">{factor.reason}</p>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="actions">
                            <h4>Recommended Actions</h4>
                            <ul>
                                {alert.recommended_actions.map((action, i) => (
                                    <li key={i}>{action}</li>
                                ))}
                            </ul>
                        </div>

                        <div className="alert-footer">
                            <button className="btn btn-secondary">
                                <Eye size={16} />
                                Investigate
                            </button>
                            <button className="btn btn-danger">
                                <Ban size={16} />
                                Block IP
                            </button>
                        </div>
                    </div>
                )}
            </div>

            <style>{`
        .alert-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          cursor: pointer;
          user-select: none;
          margin-bottom: var(--spacing-sm);
        }

        .alert-header-left,
        .alert-header-right {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
        }

        .alert-id {
          font-family: 'Courier New', monospace;
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
        }

        .alert-time {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
        }

        .alert-body {
          margin-top: var(--spacing-md);
        }

        .alert-info {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        .alert-ips {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          flex-wrap: wrap;
        }

        .ip-label {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
        }

        .ip-address {
          background: var(--color-bg-secondary);
          padding: 2px 8px;
          border-radius: var(--radius-sm);
          font-family: 'Courier New', monospace;
          font-size: var(--font-size-sm);
          color: var(--color-secondary);
        }

        .separator {
          color: var(--color-text-muted);
        }

        .threat-type {
          font-weight: 600;
          color: var(--color-text-primary);
        }

        .kill-chain {
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
        }

        .alert-expanded {
          margin-top: var(--spacing-lg);
          padding-top: var(--spacing-lg);
          border-top: 1px solid rgba(124, 58, 237, 0.1);
        }

        .explanation h4,
        .actions h4 {
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
          margin-bottom: var(--spacing-sm);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .confidence {
          font-size: var(--font-size-sm);
          margin-bottom: var(--spacing-md);
        }

        .factors {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        .factor {
          background: var(--color-bg-secondary);
          padding: var(--spacing-sm);
          border-radius: var(--radius-sm);
        }

        .factor-feature {
          font-weight: 600;
          color: var(--color-primary);
          font-family: 'Courier New', monospace;
          margin-right: var(--spacing-sm);
        }

        .factor-impact {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
        }

        .factor-reason {
          margin-top: var(--spacing-xs);
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
        }

        .actions {
          margin-top: var(--spacing-lg);
        }

        .actions ul {
          list-style: none;
          padding: 0;
        }

        .actions li {
          padding: var(--spacing-xs) 0;
          padding-left: var(--spacing-md);
          border-left: 2px solid var(--color-primary);
          margin-bottom: var(--spacing-xs);
          font-size: var(--font-size-sm);
        }

        .alert-footer {
          display: flex;
          gap: var(--spacing-sm);
          margin-top: var(--spacing-md);
        }
      `}</style>
        </div>
    );
}
