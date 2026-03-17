import type { HandleType } from "../types/pipeline";
import { SURFACE } from "../constants/theme";

/**
 * Maps each handle type to a Tailwind color class and a hex color for React Flow handles.
 */
export const HANDLE_COLORS: Record<HandleType, { bg: string; hex: string }> = {
  raw_eeg: { bg: "bg-blue-500", hex: "#3b82f6" },
  filtered_eeg: { bg: "bg-cyan-500", hex: "#06b6d4" },
  epochs: { bg: "bg-violet-500", hex: "#8b5cf6" },
  evoked: { bg: "bg-pink-500", hex: "#ec4899" },
  psd: { bg: "bg-amber-500", hex: "#f59e0b" },
  tfr: { bg: "bg-teal-500", hex: "#14b8a6" },
  plot: { bg: "bg-emerald-500", hex: "#10b981" },
  array: { bg: "bg-orange-500", hex: "#f97316" },
  scalar: { bg: "bg-rose-500", hex: "#f43f5e" },
  metrics: { bg: "bg-purple-500", hex: "#a855f7" },
  connectivity: { bg: "bg-indigo-500", hex: "#6366f1" },
  raw_fnirs: { bg: "bg-red-400", hex: "#f87171" },
  features: { bg: "bg-yellow-500", hex: "#eab308" },
};

const _catBg = `border-slate-700/50 bg-[${SURFACE.category}]`;

export const CATEGORY_COLORS: Record<string, string> = {
  "I/O": _catBg,
  Preprocessing: _catBg,
  Epoching: _catBg,
  Analysis: _catBg,
  Visualization: _catBg,
  Clinical: _catBg,
  Connectivity: _catBg,
  Statistics: _catBg,
  fNIRS: _catBg,
  Sleep: _catBg,
  BCI: _catBg,
  Compound: _catBg,
  Custom: _catBg,
};

/** Hex color for the 3px left accent stripe on node cards */
export const CATEGORY_STRIPE_HEX: Record<string, string> = {
  "I/O": "#3b82f6",
  Preprocessing: "#06b6d4",
  Epoching: "#8b5cf6",
  Analysis: "#f59e0b",
  Visualization: "#10b981",
  Clinical: "#a855f7",
  Connectivity: "#6366f1",
  Statistics: "#eab308",
  fNIRS: "#f87171",
  Sleep: "#94a3b8",
  BCI: "#84cc16",
  Compound: "#2dd4bf",
  Custom: "#f97316",
};

export const CATEGORY_HEADER_COLORS: Record<string, string> = {
  "I/O": "bg-blue-800 text-blue-100",
  Preprocessing: "bg-cyan-800 text-cyan-100",
  Epoching: "bg-violet-800 text-violet-100",
  Analysis: "bg-amber-800 text-amber-100",
  Visualization: "bg-emerald-800 text-emerald-100",
  Clinical: "bg-purple-800 text-purple-100",
  Connectivity: "bg-indigo-800 text-indigo-100",
  Statistics: "bg-yellow-800 text-yellow-100",
  Sleep: "bg-slate-700 text-slate-100",
  fNIRS: "bg-red-800 text-red-100",
  BCI: "bg-lime-800 text-lime-100",
  Compound: "bg-teal-700 text-teal-100",
  Custom: "bg-orange-800 text-orange-100",
};

export function getHandleHex(type: HandleType): string {
  return HANDLE_COLORS[type]?.hex ?? "#6b7280";
}
