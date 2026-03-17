import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("Oscilloom render crash:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col h-screen items-center justify-center bg-slate-950 text-slate-200 px-8">
          <div className="max-w-lg text-center">
            <div className="text-red-400 font-semibold text-lg mb-2">
              Something went wrong
            </div>
            <div className="text-slate-400 text-sm mb-4">
              The app crashed during rendering. This is usually caused by
              unexpected data from a pipeline run.
            </div>
            {import.meta.env.DEV ? (
              <pre className="text-xs text-red-300 bg-slate-900 border border-slate-700 rounded p-3 mb-4 text-left overflow-auto max-h-40 whitespace-pre-wrap">
                {this.state.error?.message}
                {"\n"}
                {this.state.error?.stack?.split("\n").slice(1, 6).join("\n")}
              </pre>
            ) : (
              <p className="text-sm text-slate-400 mb-4">
                Error ID: {Date.now().toString(36)} — please report this if it persists.
              </p>
            )}
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="bg-cyan-700 hover:bg-cyan-600 text-white text-sm rounded px-4 py-2 mr-2 transition-colors"
            >
              Try to Recover
            </button>
            <button
              onClick={() => window.location.reload()}
              className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm rounded px-4 py-2 transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
