import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
    title: string;
    value: string | number;
    icon?: React.ReactNode;
    trend?: {
        value: number;
        isPositive: boolean;
    };
    color?: string;
}

export function MetricCard({ title, value, icon, trend, color = 'primary' }: MetricCardProps) {
    return (
        <div className="metric-card">
            <div className="metric-header">
                <span className="metric-label">{title}</span>
                {icon && <div className="metric-icon">{icon}</div>}
            </div>
            <div className="metric-value-container">
                <div className="metric-value" style={{
                    background: `linear-gradient(135deg, var(--color-${color}), var(--color-secondary))`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text'
                }}>
                    {value}
                </div>
                {trend && (
                    <div className={`trend ${trend.isPositive ? 'positive' : 'negative'}`}>
                        {trend.isPositive ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                        <span>{Math.abs(trend.value)}%</span>
                    </div>
                )}
            </div>

            <style>{`
        .metric-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--spacing-md);
        }

        .metric-icon {
          color: var(--color-primary);
          opacity: 0.7;
        }

        .metric-value-container {
          display: flex;
          align-items: flex-end;
          gap: var(--spacing-md);
        }

        .trend {
          display: flex;
          align-items: center;
          gap: var(--spacing-xs);
          font-size: var(--font-size-sm);
          padding: 2px 8px;
          border-radius: var(--radius-sm);
        }

        .trend.positive {
          background: rgba(16, 185, 129, 0.2);
          color: var(--color-low);
        }

        .trend.negative {
          background: rgba(239, 68, 68, 0.2);
          color: var(--color-critical);
        }
      `}</style>
        </div>
    );
}
