import { useEffect } from "react";
import { createPortal } from "react-dom";

export function ImageLightbox({
  src,
  title,
  onClose,
}: {
  src: string;
  title: string;
  onClose: () => void;
}) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/85 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-w-[92vw] max-h-[92vh] rounded-xl overflow-hidden shadow-2xl border border-slate-600"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Title bar */}
        <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
          <span className="text-slate-200 text-sm font-medium">{title}</span>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-lg leading-none w-6 h-6 flex items-center justify-center rounded hover:bg-slate-700 transition-colors"
            title="Close (Esc)"
          >
            ×
          </button>
        </div>
        {/* Full-resolution image */}
        <img
          src={src}
          alt={title}
          className="block max-w-[92vw] max-h-[calc(92vh-42px)] object-contain bg-slate-900"
        />
      </div>
    </div>,
    document.body
  );
}
