import { STORAGE_KEYS } from './constants/storageKeys'

// Apply persisted theme on startup (before React render to avoid flash)
const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME);
if (savedTheme === "light") {
  document.documentElement.classList.add("light-theme");
}

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ReactFlowProvider } from '@xyflow/react'
import { ToastProvider } from './components/ui/Toast'
import { ErrorBoundary } from './components/ErrorBoundary'
import './index.css'
import Home from './pages/Home'
import Settings from './pages/Settings'
import App from './App.tsx'

function EditorPage() {
  return (
    <ReactFlowProvider>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </ReactFlowProvider>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <Routes>
          <Route path="/" element={<ErrorBoundary><Home /></ErrorBoundary>} />
          <Route path="/settings" element={<ErrorBoundary><Settings /></ErrorBoundary>} />
          <Route path="/editor" element={<EditorPage />} />
          <Route path="/editor/:workflowId" element={<EditorPage />} />
        </Routes>
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
)
