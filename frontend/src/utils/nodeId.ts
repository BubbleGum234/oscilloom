let counter = 1;

export function getNextNodeId(): string {
  return `node_${String(counter++).padStart(3, "0")}`;
}

/** After loading a pipeline, sync counter above the highest existing ID. */
export function syncNodeCounter(nodes: { id: string }[]) {
  const max = nodes.reduce((m, n) => {
    const match = n.id.match(/^node_(\d+)$/);
    return match ? Math.max(m, parseInt(match[1], 10)) : m;
  }, 0);
  if (max >= counter) counter = max + 1;
}
