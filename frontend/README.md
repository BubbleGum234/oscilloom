# Oscilloom Frontend

React + TypeScript UI for the Oscilloom EEG pipeline builder.

## Development

```bash
npm install
npm run dev
```

Opens at http://localhost:5173. Requires the backend running on port 8000.

## Building

```bash
npm run build
npm run preview   # Preview the production build
```

## Source Structure

```
src/
├── App.tsx                  # Root component
├── api/client.ts            # Typed API client for all backend endpoints
├── components/
│   ├── ErrorBoundary.tsx    # React error boundary
│   └── ui/                  # Shared UI components (Toast, etc.)
├── hooks/
│   ├── useNodeRegistry.ts   # Fetches node registry on mount
│   ├── usePipelineRunner.ts # Pipeline execution hook
│   ├── useAuditLog.ts       # In-memory audit log state
│   ├── useHistory.ts        # Undo/redo history
│   └── useRunHistory.ts     # Pipeline run history
├── pages/Home.tsx           # Main canvas page
├── store/workflowStore.ts   # Workflow state management
├── types/pipeline.ts        # TypeScript types mirroring backend models
└── utils/
    ├── handleColors.ts      # Handle type → colour mapping
    ├── nodeId.ts            # Node ID generation
    └── serializePipeline.ts # React Flow → PipelineGraph serialization
```
