// Extracted from App.tsx — global keyboard shortcut handler

import { useEffect, useRef } from "react";

export interface ShortcutHandlers {
  run: () => void;
  exportPy: () => void;
  save: () => void;
  duplicate: () => void;
  undo: () => void;
}

export function useKeyboardShortcuts(handlers: ShortcutHandlers): void {
  // Refs keep handlers fresh without re-registering the listener.
  const shortcutRefs = useRef(handlers);
  shortcutRefs.current = handlers;

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;

      switch (e.key.toLowerCase()) {
        case "z":
          if (!e.shiftKey) { e.preventDefault(); shortcutRefs.current.undo(); }
          break;
        case "s":
          e.preventDefault();
          shortcutRefs.current.save();
          break;
        case "e":
          e.preventDefault();
          shortcutRefs.current.exportPy();
          break;
        case "d":
          e.preventDefault();
          shortcutRefs.current.duplicate();
          break;
        case "r":
          e.preventDefault();
          shortcutRefs.current.run();
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
