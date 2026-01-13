import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
    Shield,
    LayoutDashboard,
    AlertTriangle,
    MessageSquare,
    Network,
    Activity
} from 'lucide-react';
import { useAlerts } from '../context/AlertContext';

const navigation = [
    { name: 'Overview', href: '/', icon: LayoutDashboard },
    { name: 'Alerts', href: '/alerts', icon: AlertTriangle },
    { name: 'Timeline', href: '/timeline', icon: Activity },
    { name: 'Attack Graph', href: '/attack-graph', icon: Network },
    { name: 'AI Copilot', href: '/copilot', icon: MessageSquare },
];

export function Sidebar() {
    const location = useLocation();
    const { isConnected } = useAlerts();

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <Shield size={32} className="logo-icon" />
                    <div>
                        <h1 className="logo-text">SENTINEL</h1>
                        <p className="logo-subtitle">AI Cyber Defense</p>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navigation.map((item) => {
                    const isActive = location.pathname === item.href;
                    const Icon = item.icon;

                    return (
                        <Link
                            key={item.name}
                            to={item.href}
                            className={`nav-item ${isActive ? 'active' : ''}`}
                        >
                            <Icon size={20} />
                            <span>{item.name}</span>
                        </Link>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                <div className="connection-status">
                    <span className={`status-indicator ${isConnected ? 'status-online' : 'status-offline'}`} />
                    <span className="status-text">
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                </div>
                <p className="version">v1.0.0-alpha</p>
            </div>

            <style>{`
        .sidebar {
          position: fixed;
          left: 0;
          top: 0;
          width: 280px;
          height: 100vh;
          background: var(--color-bg-secondary);
          border-right: 1px solid rgba(124, 58, 237, 0.2);
          display: flex;
          flex-direction: column;
          z-index: 100;
        }

        .sidebar-header {
          padding: var(--spacing-xl);
          border-bottom: 1px solid rgba(124, 58, 237, 0.1);
        }

        .logo {
          display: flex;
          align-items: center;
          gap: var(--spacing-md);
        }

        .logo-icon {
          color: var(--color-primary);
          filter: drop-shadow(0 0 10px var(--color-primary-glow));
        }

        .logo-text {
          font-size: var(--font-size-xl);
          font-weight: 700;
          background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin: 0;
        }

        .logo-subtitle {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
          margin: 0;
        }

        .sidebar-nav {
          flex: 1;
          padding: var(--spacing-lg) 0;
          overflow-y: auto;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--spacing-md);
          padding: var(--spacing-md) var(--spacing-xl);
          color: var(--color-text-secondary);
          text-decoration: none;
          transition: all var(--transition-fast);
          position: relative;
        }

        .nav-item::before {
          content: '';
          position: absolute;
          left: 0;
          top: 50%;
          transform: translateY(-50%);
          width: 3px;
          height: 0;
          background: linear-gradient(180deg, var(--color-primary), var(--color-secondary));
          transition: height var(--transition-base);
        }

        .nav-item:hover {
          background: rgba(124, 58, 237, 0.1);
          color: var(--color-text-primary);
        }

        .nav-item.active {
          color: var(--color-primary);
          background: rgba(124, 58, 237, 0.15);
        }

        .nav-item.active::before {
          height: 100%;
        }

        .sidebar-footer {
          padding: var(--spacing-lg) var(--spacing-xl);
          border-top: 1px solid rgba(124, 58, 237, 0.1);
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          margin-bottom: var(--spacing-sm);
        }

        .status-text {
          font-size: var(--font-size-sm);
          color: var(--color-text-secondary);
        }

        .version {
          font-size: var(--font-size-xs);
          color: var(--color-text-muted);
          text-align: center;
          margin: 0;
        }
      `}</style>
        </aside>
    );
}
