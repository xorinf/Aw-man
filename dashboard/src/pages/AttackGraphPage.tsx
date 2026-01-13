import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import api, { type AttackGraph } from '../api/client';

export function AttackGraphPage() {
    const containerRef = useRef<HTMLDivElement>(null);
    const [graph, setGraph] = useState<AttackGraph | null>(null);

    useEffect(() => {
        api.getAttackGraph().then(setGraph);
    }, []);

    useEffect(() => {
        if (!containerRef.current || !graph) return;

        const cy = cytoscape({
            container: containerRef.current,
            elements: [
                ...graph.nodes.map(node => ({
                    data: { id: node.id, label: node.label, type: node.type, compromised: node.compromised }
                })),
                ...graph.edges.map(edge => ({
                    data: { source: edge.source, target: edge.target, label: edge.label }
                }))
            ],
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#7c3aed',
                        'label': 'data(label)',
                        'color': '#f8fafc',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'width': '60px',
                        'height': '60px',
                        'border-width': '2px',
                        'border-color': '#06b6d4'
                    }
                },
                {
                    selector: 'node[type="host"]',
                    style: {
                        'shape': 'roundrectangle',
                        'background-color': '#0ea5e9'
                    }
                },
                {
                    selector: 'node[compromised]',
                    style: {
                        'background-color': '#ef4444',
                        'border-color': '#fca5a5'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#7c3aed',
                        'target-arrow-color': '#7c3aed',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '10px',
                        'color': '#cbd5e1',
                        'text-rotation': 'autorotate'
                    }
                }
            ],
            layout: {
                name: 'circle',
                padding: 50
            }
        });

        return () => cy.destroy();
    }, [graph]);

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">Attack Graph</h1>
                <p className="page-subtitle">Visualize attack paths and compromised assets</p>
            </div>

            <div className="card graph-container">
                <div ref={containerRef} className="cytoscape-container" />

                <div className="graph-legend">
                    <div className="legend-item">
                        <div className="legend-dot" style={{ background: '#0ea5e9' }} />
                        <span>Host</span>
                    </div>
                    <div className="legend-item">
                        <div className="legend-dot" style={{ background: '#7c3aed' }} />
                        <span>Technique</span>
                    </div>
                    <div className="legend-item">
                        <div className="legend-dot" style={{ background: '#ef4444' }} />
                        <span>Compromised</span>
                    </div>
                </div>
            </div>

            <style>{`
        .graph-container {
          position: relative;
          height: 600px;
        }

        .cytoscape-container {
          width: 100%;
          height: 100%;
          background: var(--color-bg-primary);
          border-radius: var(--radius-lg);
        }

        .graph-legend {
          position: absolute;
          top: var(--spacing-lg);
          right: var(--spacing-lg);
          background: var(--color-surface);
          padding: var(--spacing-md);
          border-radius: var(--radius-md);
          display: flex;
          flex-direction: column;
          gap: var(--spacing-sm);
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          font-size: var(--font-size-sm);
        }

        .legend-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid rgba(255, 255, 255, 0.3);
        }
      `}</style>
        </div>
    );
}
