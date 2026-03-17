import { useState, useCallback } from "react";
import { saveCustomNode } from "../../api/client";

interface SaveCustomNodeModalProps {
  code: string;
  timeout: number;
  onClose: () => void;
}

export function SaveCustomNodeModal({ code, timeout, onClose }: SaveCustomNodeModalProps) {
  const [saveName, setSaveName] = useState("");
  const [saveDesc, setSaveDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<string | null>(null);

  const handleSave = useCallback(async () => {
    if (!saveName.trim()) return;
    setSaving(true);
    setSaveResult(null);
    try {
      await saveCustomNode(saveName.trim(), saveDesc.trim(), code, timeout);
      setSaveResult("Saved! Reload palette to see it.");
      onClose();
    } catch (err) {
      setSaveResult(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [saveName, saveDesc, code, timeout, onClose]);

  return (
    <>
      <div className="space-y-1 bg-slate-900/80 rounded p-1.5 border border-slate-700 nodrag">
        <input
          type="text"
          placeholder="Node name..."
          value={saveName}
          onChange={(e) => setSaveName(e.target.value)}
          className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-0.5 text-slate-200 text-[10px] focus:outline-none"
        />
        <input
          type="text"
          placeholder="Description (optional)"
          value={saveDesc}
          onChange={(e) => setSaveDesc(e.target.value)}
          className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-0.5 text-slate-200 text-[10px] focus:outline-none"
        />
        <div className="flex gap-1">
          <button
            onClick={handleSave}
            disabled={saving || !saveName.trim()}
            className="flex-1 text-[10px] bg-orange-700 hover:bg-orange-600 disabled:opacity-50 text-white rounded px-2 py-0.5 transition-colors"
          >
            {saving ? "Saving..." : "Save"}
          </button>
          <button
            onClick={onClose}
            className="text-[10px] text-slate-400 hover:text-slate-200 px-2"
          >
            Cancel
          </button>
        </div>
      </div>
      {saveResult && (
        <div className={`text-[9px] mt-0.5 px-1 ${saveResult.includes("Saved") ? "text-green-400" : "text-red-400"}`}>
          {saveResult}
        </div>
      )}
    </>
  );
}
