import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import { createPortal } from "react-dom";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ToastVariant = "success" | "error" | "warning" | "info";

interface Toast {
  id: number;
  message: string;
  variant: ToastVariant;
  exiting?: boolean;
}

interface ToastContextValue {
  toast: (message: string, variant?: ToastVariant) => void;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

// ---------------------------------------------------------------------------
// Provider + renderer
// ---------------------------------------------------------------------------

const DURATION_MS = 3500;
const EXIT_MS = 300;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(0);

  const toast = useCallback((message: string, variant: ToastVariant = "info") => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { id, message, variant }]);

    // Start exit animation, then remove
    setTimeout(() => {
      setToasts((prev) =>
        prev.map((t) => (t.id === id ? { ...t, exiting: true } : t))
      );
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, EXIT_MS);
    }, DURATION_MS);
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {createPortal(<ToastContainer toasts={toasts} />, document.body)}
    </ToastContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const VARIANT_STYLES: Record<ToastVariant, string> = {
  success: "bg-cyan-900 border-cyan-600 text-cyan-100",
  error: "bg-red-900 border-red-600 text-red-100",
  warning: "bg-amber-900 border-amber-600 text-amber-100",
  info: "bg-slate-800 border-slate-600 text-slate-100",
};

const VARIANT_ICONS: Record<ToastVariant, string> = {
  success: "\u2713",
  error: "\u2717",
  warning: "\u26A0",
  info: "\u2139",
};

function ToastContainer({ toasts }: { toasts: Toast[] }) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-[10000] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}

function ToastItem({ toast: t }: { toast: Toast }) {
  const [visible, setVisible] = useState(false);

  // Trigger enter animation on mount
  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  const isExiting = t.exiting;

  return (
    <div
      className={`
        pointer-events-auto flex items-center gap-2 px-4 py-2.5 rounded-lg border shadow-lg
        text-xs font-medium max-w-sm
        transition-all duration-300 ease-out
        ${VARIANT_STYLES[t.variant]}
        ${visible && !isExiting ? "opacity-100 translate-x-0" : "opacity-0 translate-x-8"}
      `}
    >
      <span className="text-sm">{VARIANT_ICONS[t.variant]}</span>
      <span>{t.message}</span>
    </div>
  );
}
