import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import { AlertCard } from '../components/AlertCard';
import { useAlerts } from '../context/AlertContext';

export function Alerts() {
    const { alerts } = useAlerts();
    const [searchTerm, setSearchTerm] = useState('');
    const [severityFilter, setSeverityFilter] = useState<string>('all');

    const filteredAlerts = alerts.filter(alert => {
        const matchesSearch =
            alert.source_ip.includes(searchTerm) ||
            alert.destination_ip.includes(searchTerm) ||
            alert.alert_id.toLowerCase().includes(searchTerm.toLowerCase());

        const matchesSeverity =
            severityFilter === 'all' || alert.verdict === severityFilter;

        return matchesSearch && matchesSeverity;
    });

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">Threat Alerts</h1>
                <p className="page-subtitle">Monitor and investigate security threats</p>
            </div>

            <div className="alert-controls">
                <div className="search-box">
                    <Search size={18} />
                    <input
                        type="text"
                        className="input search-input"
                        placeholder="Search by IP or Alert ID..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>

                <div className="filter-group">
                    <Filter size={18} />
                    <select
                        className="input filter-select"
                        value={severityFilter}
                        onChange={(e) => setSeverityFilter(e.target.value)}
                    >
                        <option value="all">All Severities</option>
                        <option value="CRITICAL">Critical</option>
                        <option value="HIGH">High</option>
                        <option value="MEDIUM">Medium</option>
                        <option value="LOW">Low</option>
                    </select>
                </div>
            </div>

            <div className="alert-stats">
                <div className="stat">
                    <span className="stat-value">{filteredAlerts.length}</span>
                    <span className="stat-label">Total</span>
                </div>
                <div className="stat critical">
                    <span className="stat-value">{filteredAlerts.filter(a => a.verdict === 'CRITICAL').length}</span>
                    <span className="stat-label">Critical</span>
                </div>
                <div className="stat high">
                    <span className="stat-value">{filteredAlerts.filter(a => a.verdict === 'HIGH').length}</span>
                    <span className="stat-label">High</span>
                </div>
            </div>

            <div className="alerts-list">
                {filteredAlerts.map(alert => (
                    <AlertCard key={alert.alert_id} alert={alert} />
                ))}
                {filteredAlerts.length === 0 && (
                    <div className="empty-state">
                        <p>No alerts match your filters</p>
                    </div>
                )}
            </div>

            <style>{`
        .alert-controls {
          display: flex;
          gap: var(--spacing-md);
          margin-bottom: var(--spacing-xl);
        }

        .search-box,
        .filter-group {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          flex: 1;
        }

        .search-input,
        .filter-select {
          flex: 1;
        }

        .alert-stats {
          display: flex;
          gap: var(--spacing-lg);
          margin-bottom: var(--spacing-xl);
        }

        .stat {
          flex: 1;
          background: var(--color-surface);
          padding: var(--spacing-lg);
          border-radius: var(--radius-lg);
          border-left: 3px solid var(--color-primary);
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .stat.critical {
          border-left-color: var(--color-critical);
        }

        .stat.high {
          border-left-color: var(--color-high);
        }

        .stat-value {
          font-size: var(--font-size-2xl);
          font-weight: 700;
          color: var(--color-text-primary);
        }

        .stat-label {
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .alerts-list {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
        }
      `}</style>
        </div>
    );
}
