import { useState, useCallback } from "react";
import type { NodeRegistry, NodeTypeDescriptor } from "../../types/pipeline";
import { CATEGORY_HEADER_COLORS, CATEGORY_COLORS } from "../../utils/handleColors";
import { deleteCompound, deleteCustomNode } from "../../api/client";
import { Trash2, Heart } from "lucide-react";

interface NodePaletteProps {
  registry: NodeRegistry;
  onRegistryChanged?: () => void;
}

function isUserCreatedNode(nodeType: string): boolean {
  return nodeType.startsWith("c_") || nodeType.startsWith("custom__");
}

function PaletteCard({
  descriptor,
  onDragStart,
  onDelete,
}: {
  descriptor: NodeTypeDescriptor;
  onDragStart: (e: React.DragEvent, descriptor: NodeTypeDescriptor) => void;
  onDelete?: (descriptor: NodeTypeDescriptor) => void;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const borderClass = CATEGORY_COLORS[descriptor.category] ?? "border-gray-600 bg-gray-900";
  const deletable = onDelete && isUserCreatedNode(descriptor.node_type);

  return (
    <div
      draggable={!confirmingDelete}
      onDragStart={(e) => onDragStart(e, descriptor)}
      className={`
        cursor-grab active:cursor-grabbing
        rounded border-2 px-3 py-2 select-none
        hover:brightness-125 transition-all relative group
        ${borderClass}
      `}
      title={descriptor.description}
    >
      <div className="flex items-center justify-between gap-1">
        <div className="font-medium text-slate-200 text-xs truncate flex items-center gap-1">
          {descriptor.display_name}
          {descriptor.node_type === "ant_loader" && (
            <Heart className="w-2.5 h-2.5 fill-slate-400 text-slate-400 flex-shrink-0" />
          )}
        </div>
        {deletable && !confirmingDelete && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setConfirmingDelete(true);
            }}
            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-950/40 transition-all flex-shrink-0"
            title={`Delete ${descriptor.display_name}`}
          >
            <Trash2 className="w-3 h-3" />
          </button>
        )}
      </div>
      {confirmingDelete ? (
        <div className="flex items-center gap-1.5 mt-1">
          <span className="text-[10px] text-red-400">Delete?</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete!(descriptor);
              setConfirmingDelete(false);
            }}
            className="text-[10px] px-1.5 py-0.5 rounded bg-red-600 hover:bg-red-500 text-white font-medium transition-colors"
          >
            Yes
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setConfirmingDelete(false);
            }}
            className="text-[10px] px-1.5 py-0.5 rounded border border-slate-600 text-slate-400 hover:bg-slate-800 transition-colors"
          >
            No
          </button>
        </div>
      ) : (
        descriptor.parameters.length > 0 && (
          <div className="text-slate-500 text-[10px] mt-0.5">
            {descriptor.parameters.length} param{descriptor.parameters.length !== 1 ? "s" : ""}
          </div>
        )
      )}
    </div>
  );
}

export function NodePalette({ registry, onRegistryChanged }: NodePaletteProps) {
  const [search, setSearch] = useState("");

  const handleDragStart = (e: React.DragEvent, descriptor: NodeTypeDescriptor) => {
    e.dataTransfer.setData("application/oscilloom-node", descriptor.node_type);
    e.dataTransfer.effectAllowed = "copy";
  };

  const handleDeleteNode = useCallback(async (descriptor: NodeTypeDescriptor) => {
    try {
      if (descriptor.node_type.startsWith("c_")) {
        await deleteCompound(descriptor.node_type);
      } else if (descriptor.node_type.startsWith("custom__")) {
        const slug = descriptor.node_type.slice(8);
        await deleteCustomNode(slug);
      }
      onRegistryChanged?.();
    } catch (err) {
      console.error("Failed to delete node:", err);
    }
  }, [onRegistryChanged]);

  const query = search.trim().toLowerCase();

  // When searching: flat filtered list; when empty: category-grouped view.
  const filtered = query
    ? Object.values(registry).filter(
        (desc) =>
          desc.display_name.toLowerCase().includes(query) ||
          desc.category.toLowerCase().includes(query) ||
          (desc.tags ?? []).some((t) => t.toLowerCase().includes(query))
      )
    : null;

  const byCategory: Record<string, NodeTypeDescriptor[]> = {};
  for (const desc of Object.values(registry)) {
    if (!byCategory[desc.category]) byCategory[desc.category] = [];
    byCategory[desc.category].push(desc);
  }
  const categoryOrder = ["I/O", "Preprocessing", "Epoching", "Analysis", "Visualization", "Compound"];
  const sortedCategories = [
    ...categoryOrder.filter((c) => byCategory[c]),
    ...Object.keys(byCategory).filter((c) => !categoryOrder.includes(c)),
  ];

  return (
    <div className="w-full h-full border-r border-slate-700 bg-[#131825] flex flex-col overflow-hidden">
      <div className="px-3 py-2.5 border-b border-slate-700 flex-shrink-0">
        <h2 className="text-slate-400 font-semibold text-[11px] uppercase tracking-wide">Nodes</h2>
        <div className="relative mt-1.5">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search nodes…"
            className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs placeholder-slate-500 focus:outline-none focus:border-slate-400 pr-6"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 text-xs leading-none"
              title="Clear search"
            >
              ×
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {filtered !== null ? (
          <>
            {filtered.length === 0 ? (
              <p className="text-slate-600 text-xs text-center mt-6 px-2">
                No nodes match &quot;{search}&quot;
              </p>
            ) : (
              <div className="space-y-1">
                {filtered.map((desc) => (
                  <PaletteCard
                    key={desc.node_type}
                    descriptor={desc}
                    onDragStart={handleDragStart}
                    onDelete={handleDeleteNode}
                  />
                ))}
              </div>
            )}
          </>
        ) : (
          sortedCategories.map((category) => (
            <div key={category}>
              <div
                className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-1 rounded mb-1.5 ${
                  CATEGORY_HEADER_COLORS[category] ?? "bg-slate-700 text-slate-300"
                }`}
              >
                {category}
              </div>
              <div className="space-y-1">
                {byCategory[category].map((desc) => (
                  <PaletteCard
                    key={desc.node_type}
                    descriptor={desc}
                    onDragStart={handleDragStart}
                    onDelete={handleDeleteNode}
                  />
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
