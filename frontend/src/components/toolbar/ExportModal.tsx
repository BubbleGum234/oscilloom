import { useState, useEffect, useCallback } from 'react';
import { X, Copy, Download, FileText, BookOpen, Package, Check } from 'lucide-react';
import { exportPipeline, generateMethods, exportPackage } from '../../api/client';
import type { PipelineGraph } from '../../types/pipeline';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  pipeline: PipelineGraph;
  auditLog: unknown[];
}

type Tab = 'script' | 'methods' | 'package';

export default function ExportModal({ isOpen, onClose, sessionId, pipeline, auditLog }: ExportModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>('script');
  const [scriptContent, setScriptContent] = useState<string>('');
  const [scriptFilename, setScriptFilename] = useState<string>('pipeline.py');
  const [methodsContent, setMethodsContent] = useState<string>('');
  const [wordCount, setWordCount] = useState(0);
  const [citations, setCitations] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch content when the modal opens or tab changes
  useEffect(() => {
    if (!isOpen) return;

    if (activeTab === 'script' && !scriptContent) {
      setLoading(true);
      setError(null);
      exportPipeline(sessionId, pipeline, auditLog as object[])
        .then(({ script, filename }) => {
          setScriptContent(script);
          setScriptFilename(filename);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => setLoading(false));
    }

    if (activeTab === 'methods' && !methodsContent) {
      setLoading(true);
      setError(null);
      generateMethods(sessionId, pipeline)
        .then(({ methods_section, word_count, citations: cites }) => {
          setMethodsContent(methods_section);
          setWordCount(word_count);
          setCitations(cites);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen, activeTab, sessionId, pipeline, auditLog, scriptContent, methodsContent]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setScriptContent('');
      setMethodsContent('');
      setWordCount(0);
      setCitations([]);
      setCopied(false);
      setError(null);
      setActiveTab('script');
    }
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  const handleCopy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, []);

  const handleDownloadScript = useCallback(() => {
    const blob = new Blob([scriptContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = scriptFilename;
    a.click();
    URL.revokeObjectURL(url);
  }, [scriptContent, scriptFilename]);

  const handleDownloadPackage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const blob = await exportPackage(sessionId, pipeline, auditLog);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${pipeline.metadata?.name?.replace(/\s+/g, '_') || 'pipeline'}_package.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [sessionId, pipeline, auditLog]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-slate-800 rounded-xl shadow-2xl w-[700px] max-h-[80vh] flex flex-col border border-slate-700"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">Export Pipeline</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          {([
            { id: 'script' as Tab, label: 'Python Script', icon: FileText },
            { id: 'methods' as Tab, label: 'Methods', icon: BookOpen },
            { id: 'package' as Tab, label: 'Package', icon: Package },
          ]).map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setCopied(false); }}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <tab.icon size={14} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {/* Script Tab */}
          {activeTab === 'script' && (
            <div>
              {loading ? (
                <div className="text-slate-400 text-sm">Generating script...</div>
              ) : error ? (
                <div className="text-red-400 text-sm">{error}</div>
              ) : (
                <>
                  <div className="flex gap-2 mb-3">
                    <button
                      onClick={() => handleCopy(scriptContent)}
                      className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded text-xs transition-colors"
                    >
                      {copied ? <Check size={12} /> : <Copy size={12} />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <button
                      onClick={handleDownloadScript}
                      className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs transition-colors"
                    >
                      <Download size={12} /> Download .py
                    </button>
                  </div>
                  <pre className="bg-slate-900 text-blue-300 p-4 rounded-lg text-xs font-mono overflow-auto max-h-96 leading-relaxed whitespace-pre">
                    {scriptContent}
                  </pre>
                </>
              )}
            </div>
          )}

          {/* Methods Tab */}
          {activeTab === 'methods' && (
            <div>
              {loading ? (
                <div className="text-slate-400 text-sm">Generating methods section...</div>
              ) : error ? (
                <div className="text-red-400 text-sm">{error}</div>
              ) : (
                <>
                  <div className="flex items-center gap-3 mb-3">
                    <button
                      onClick={() => handleCopy(methodsContent)}
                      className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded text-xs transition-colors"
                    >
                      {copied ? <Check size={12} /> : <Copy size={12} />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <span className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded">
                      {wordCount} words
                    </span>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                    {methodsContent || 'No methods section available.'}
                  </div>
                  {citations.length > 0 && (
                    <div className="mt-3 text-xs text-slate-400">
                      <span className="font-medium">Citations:</span> {citations.join('; ')}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Package Tab */}
          {activeTab === 'package' && (
            <div>
              <p className="text-sm text-slate-400 mb-3">
                Download a reproducibility package containing all pipeline artifacts.
              </p>
              <div className="space-y-2 mb-4">
                {['pipeline.py', 'pipeline.json', 'requirements.txt', 'README.md'].map((file) => (
                  <div
                    key={file}
                    className="flex items-center gap-2 bg-slate-900 px-3 py-2 rounded text-sm text-slate-300"
                  >
                    <FileText size={14} className="text-slate-500" />
                    {file}
                  </div>
                ))}
              </div>
              {error && (
                <div className="text-red-400 text-sm mb-3">{error}</div>
              )}
              <button
                onClick={handleDownloadPackage}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:text-slate-400 text-white rounded text-sm font-medium transition-colors"
              >
                <Package size={14} />
                {loading ? 'Preparing...' : 'Download .zip'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
