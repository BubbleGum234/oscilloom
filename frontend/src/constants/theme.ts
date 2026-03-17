// Centralized design tokens — single source of truth for all colors
// Replaces 30+ scattered hex values across the codebase

export const SURFACE = {
  /** Page-level background (darkest) */
  page: "#0f1117",
  /** Main canvas background */
  canvas: "#1e2738",
  /** Toolbar background */
  toolbar: "#171e2d",
  /** Side panel scroll area */
  panel: "#151b28",
  /** Side panel sticky headers */
  panelHeader: "#1a2133",
  /** Node body background */
  node: "#1e293b",
  /** Node header stripe area */
  nodeHeader: "#1e2740",
  /** Category background in palette */
  category: "#1a2030",
  /** Minimap background */
  minimap: "#1e293b",
  /** Modal/dialog overlay content */
  modal: "#1e293b",
  /** Elevated surface (tooltips, dropdowns) */
  elevated: "#2a3348",
} as const;

export const BORDER = {
  subtle: "#334155",
  default: "#475569",
} as const;
