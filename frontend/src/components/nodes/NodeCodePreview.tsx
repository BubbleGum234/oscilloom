import { useState, useCallback } from "react";
import { Code2 } from "lucide-react";
import { getNodeCode } from "../../api/client";

interface NodeCodePreviewProps {
  nodeType: string;
  parameters: Record<string, unknown>;
}

export function NodeCodePreview({ nodeType, parameters }: NodeCodePreviewProps) {
  const [showCode, setShowCode] = useState(false);
  const [codeData, setCodeData] = useState<{ code: string | null; docs_url: string | null } | null>(null);
  const [codeLoading, setCodeLoading] = useState(false);

  const handleToggleCode = useCallback(async () => {
    if (showCode) {
      setShowCode(false);
      return;
    }
    setCodeLoading(true);
    try {
      const result = await getNodeCode(nodeType ?? "", parameters || {});
      setCodeData(result);
      setShowCode(true);
    } catch {
      setCodeData({ code: "// Failed to load code preview", docs_url: null });
      setShowCode(true);
    } finally {
      setCodeLoading(false);
    }
  }, [showCode, nodeType, parameters]);

  return { showCode, codeLoading, codeData, handleToggleCode };
}

export function CodeToggleButton({
  showCode,
  onClick,
}: {
  showCode: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      className={`p-0.5 rounded transition-colors nodrag ${
        showCode
          ? "text-blue-400 bg-blue-400/20"
          : "text-slate-500 hover:text-slate-300"
      }`}
      title="View MNE code"
    >
      <Code2 size={12} />
    </button>
  );
}

export function CodePreviewPanel({
  showCode,
  codeLoading,
  codeData,
}: {
  showCode: boolean;
  codeLoading: boolean;
  codeData: { code: string | null; docs_url: string | null } | null;
}) {
  if (!showCode) return null;

  return (
    <div className="border-t border-slate-600/50 bg-slate-900/80 px-2 py-1.5">
      {codeLoading ? (
        <div className="text-xs text-slate-500 font-mono">Loading...</div>
      ) : codeData?.code ? (
        <>
          <pre className="text-[10px] leading-relaxed font-mono text-green-300 whitespace-pre-wrap overflow-x-auto max-h-32 overflow-y-auto">
            {codeData.code}
          </pre>
          {codeData.docs_url && (
            <a
              href={codeData.docs_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[9px] text-blue-400 hover:text-blue-300 hover:underline mt-1 inline-block"
            >
              MNE Docs &rarr;
            </a>
          )}
        </>
      ) : (
        <div className="text-[10px] text-slate-500 font-mono italic">No code template available</div>
      )}
    </div>
  );
}
