import React, { useEffect, useState } from 'react';
import { Shield, AlertTriangle, Activity, TrendingUp } from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { AlertCard } from '../components/AlertCard';
import { useAlerts } from '../context/AlertContext';
import api from '../api/client';

export function Overview() {
    const { alerts } = useAlerts();
    const [mitreCoverage, setMitreCoverage] = useState<any>(null);

    useEffect(() => {
        api.getMITRECoverage().then(setMitreCoverage);
    }, []);

    const criticalCount = alerts.filter(a => a.verdict === 'CRITICAL').length;
    const highCount = alerts.filter(a => a.verdict === 'HIGH').length;
    const recentAlerts = alerts.slice(0, 5);

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">Security Overview</h1>
                <p className="page-subtitle">Real-time threat monitoring and analysis</p>
            </div>

            <div className="grid grid-4">
                <MetricCard
                    title="Total Alerts"
                    value={alerts.length}
                    icon={<AlertTriangle size={24} />}
                    trend={{ value: 12, isPositive: false }}
                />
                <MetricCard
                    title="Critical Threats"
                    value={criticalCount}
                    icon={<Shield size={24} />}
                    color="critical"
                />
                <MetricCard
                    title="High Priority"
                    value={highCount}
                    icon={<Activity size={24} />}
                    color="high"
                />
                <MetricCard
                    title="Detection Rate"
                    value={mitreCoverage ? `${(mitreCoverage.detection_rate * 100).toFixed(0)}%` : '...'}
                    icon={<TrendingUp size={24} />}
                    trend={{ value: 5, isPositive: true }}
                />
            </div>

            <div className="content-section">
                <div className="section-header">
                    <h2>Recent Alerts</h2>
                    <a href="/alerts" className="link">View all →</a>
                </div>
                <div className="alerts-feed">
                    {recentAlerts.length > 0 ? (
                        recentAlerts.map(alert => (
                            <AlertCard key={alert.alert_id} alert={alert} />
                        ))
                    ) : (
                        <div className="empty-state">
                            <Shield size={48} style={{ color: 'var(--color-primary)', opacity: 0.5 }} />
                            <p>No alerts detected. Monitoring is active.</p>
                        </div>
                    )}
                </div>
            </div>

            <style>{`
        .page-container {
          animation: fadeIn var(--transition-base) ease-out;
        }

        .page-header {
          margin-bottom: var(--spacing-2xl);
        }

        .page-title {
          font-size: var(--font-size-3xl);
          font-weight: 700;
          background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: var(--spacing-xs);
        }

        .page-subtitle {
          color: var(--color-text-secondary);
          font-size: var(--font-size-lg);
        }

        .content-section {
          margin-top: var(--spacing-2xl);
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--spacing-lg);
        }

        .section-header h2 {
          font-size: var(--font-size-xl);
          font-weight: 600;
        }

        .link {
          color: var(--color-primary);
          text-decoration: none;
          transition: color var(--transition-fast);
        }

        .link:hover {
          color: var(--color-secondary);
        }

        .alerts-feed {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
        }

        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--spacing-2xl);
          color: var(--color-text-muted);
          text-align: center;
        }
      `}</style>
        </div>
    );
}
