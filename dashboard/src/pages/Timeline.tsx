import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useAlerts } from '../context/AlertContext';

export function Timeline() {
    const { alerts } = useAlerts();

    // Group alerts by time for chart
    const chartData = alerts.slice(0, 20).reverse().map((alert, i) => ({
        index: i,
        time: new Date(alert.timestamp).toLocaleTimeString(),
        severity: alert.verdict === 'CRITICAL' ? 4 : alert.verdict === 'HIGH' ? 3 : alert.verdict === 'MEDIUM' ? 2 : 1
    }));

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">Attack Timeline</h1>
                <p className="page-subtitle">Visualize threats over time</p>
            </div>

            <div className="card chart-container">
                <h3 className="chart-title">Alert Severity Timeline</h3>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(124, 58, 237, 0.2)" />
                        <XAxis
                            dataKey="time"
                            stroke="var(--color-text-muted)"
                            style={{ fontSize: '12px' }}
                        />
                        <YAxis
                            stroke="var(--color-text-muted)"
                            style={{ fontSize: '12px' }}
                        />
                        <Tooltip
                            contentStyle={{
                                background: 'var(--color-surface)',
                                border: '1px solid rgba(124, 58, 237, 0.3)',
                                borderRadius: 'var(--radius-md)'
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="severity"
                            stroke="url(#colorGradient)"
                            strokeWidth={3}
                            dot={{ fill: 'var(--color-primary)', r: 4 }}
                        />
                        <defs>
                            <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stopColor="var(--color-primary)" />
                                <stop offset="100%" stopColor="var(--color-secondary)" />
                            </linearGradient>
                        </defs>
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="timeline-list">
                {alerts.map((alert, index) => (
                    <div key={alert.alert_id} className="timeline-item">
                        <div className="timeline-marker">
                            <div className={`marker-dot ${alert.verdict.toLowerCase()}`} />
                            {index < alerts.length - 1 && <div className="timeline-line" />}
                        </div>
                        <div className="timeline-content">
                            <div className="timeline-time">{new Date(alert.timestamp).toLocaleString()}</div>
                            <div className="timeline-details">
                                <span className={`badge badge-${alert.verdict.toLowerCase()}`}>{alert.verdict}</span>
                                <span className="timeline-threat">{alert.threat_type || 'Unknown Threat'}</span>
                            </div>
                            <div className="timeline-ips">
                                {alert.source_ip} → {alert.destination_ip}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <style>{`
        .chart-container {
          margin-bottom: var(--spacing-2xl);
        }

        .chart-title {
          font-size: var(--font-size-lg);
          margin-bottom: var(--spacing-lg);
          color: var(--color-text-secondary);
        }

        .timeline-list {
          display: flex;
          flex-direction: column;
        }

        .timeline-item {
          display: flex;
          gap: var(--spacing-lg);
        }

        .timeline-marker {
          display: flex;
          flex-direction: column;
          align-items: center;
          position: relative;
        }

        .marker-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: var(--color-primary);
          box-shadow: 0 0 10px var(--color-primary-glow);
          position: relative;
          z-index: 1;
        }

        .marker-dot.critical {
          background: var(--color-critical);
          box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }

        .marker-dot.high {
          background: var(--color-high);
        }

        .timeline-line {
          width: 2px;
          flex: 1;
          background: linear-gradient(180deg, var(--color-primary), transparent);
          margin-top: 4px;
        }

        .timeline-content {
          flex: 1;
          padding-bottom: var(--spacing-xl);
        }

        .timeline-time {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
          margin-bottom: var(--spacing-xs);
        }

        .timeline-details {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          margin-bottom: var(--spacing-xs);
        }

        .timeline-threat {
          font-weight: 500;
        }

        .timeline-ips {
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
          font-family: 'Courier New', monospace;
        }
      `}</style>
        </div>
    );
}
