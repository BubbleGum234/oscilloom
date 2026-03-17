// Extracted from App.tsx — resizable side panel logic
import { useState, useRef, useEffect, useCallback } from "react";
import { STORAGE_KEYS } from "../constants/storageKeys";

export interface UsePanelResizeReturn {
  leftPanelWidth: number;
  rightPanelWidth: number;
  startLeftDrag: (e: React.MouseEvent) => void;
  startRightDrag: (e: React.MouseEvent) => void;
}

export function usePanelResize(): UsePanelResizeReturn {
  const [leftPanelWidth, setLeftPanelWidth] = useState<number>(() => {
    try { const v = parseInt(localStorage.getItem(STORAGE_KEYS.LEFT_PANEL_WIDTH) || ""); return v >= 140 && v <= 400 ? v : 208; } catch { return 208; }
  });
  const [rightPanelWidth, setRightPanelWidth] = useState<number>(() => {
    try { const v = parseInt(localStorage.getItem(STORAGE_KEYS.RIGHT_PANEL_WIDTH) || ""); return v >= 200 && v <= 600 ? v : 320; } catch { return 320; }
  });

  const leftDrag = useRef(false);
  const rightDrag = useRef(false);

  // Persist to localStorage
  useEffect(() => { try { localStorage.setItem(STORAGE_KEYS.LEFT_PANEL_WIDTH, String(leftPanelWidth)); } catch {} }, [leftPanelWidth]);
  useEffect(() => { try { localStorage.setItem(STORAGE_KEYS.RIGHT_PANEL_WIDTH, String(rightPanelWidth)); } catch {} }, [rightPanelWidth]);

  // Global mousemove / mouseup handlers for drag resize
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (leftDrag.current) {
        setLeftPanelWidth(Math.max(140, Math.min(400, e.clientX)));
      }
      if (rightDrag.current) {
        setRightPanelWidth(Math.max(200, Math.min(600, window.innerWidth - e.clientX)));
      }
    };
    const onUp = () => {
      if (leftDrag.current || rightDrag.current) {
        leftDrag.current = false;
        rightDrag.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  const startLeftDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    leftDrag.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  const startRightDrag = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    rightDrag.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  return { leftPanelWidth, rightPanelWidth, startLeftDrag, startRightDrag };
}
