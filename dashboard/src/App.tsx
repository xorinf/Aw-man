import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AlertProvider } from './context/AlertContext';
import { Sidebar } from './components/Sidebar';
import { Overview } from './pages/Overview';
import { Alerts } from './pages/Alerts';
import { Timeline } from './pages/Timeline';
import { AttackGraphPage } from './pages/AttackGraphPage';
import { Copilot } from './pages/Copilot';
import './styles/index.css';

function App() {
    return (
        <AlertProvider>
            <BrowserRouter>
                <div className="app-container">
                    <Sidebar />
                    <main className="main-content">
                        <Routes>
                            <Route path="/" element={<Overview />} />
                            <Route path="/alerts" element={<Alerts />} />
                            <Route path="/timeline" element={<Timeline />} />
                            <Route path="/attack-graph" element={<AttackGraphPage />} />
                            <Route path="/copilot" element={<Copilot />} />
                        </Routes>
                    </main>
                </div>
            </BrowserRouter>
        </AlertProvider>
    );
}

export default App;
