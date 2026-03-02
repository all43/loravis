import React, { useState, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";

const Adaptive = lazy(() => import("../loravis-adaptive.jsx"));
const MultiStrategy = lazy(() => import("../loravis.jsx"));
const Advanced = lazy(() => import("../loravis-advanced.jsx"));

const TABS = [
  { id: "adaptive", label: "Adaptive Selector", Component: Adaptive },
  { id: "multi", label: "Multi-Strategy", Component: MultiStrategy },
  { id: "advanced", label: "Advanced Deep-Dive", Component: Advanced },
];

function Shell() {
  const [tab, setTab] = useState("adaptive");
  const current = TABS.find((t) => t.id === tab);

  return (
    <>
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          gap: 0,
          background: "#010409",
          borderBottom: "1px solid #21262d",
          padding: "0 20px",
          position: "sticky",
          top: 0,
          zIndex: 200,
        }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: "10px 20px",
              background: "transparent",
              color: tab === t.id ? "#58a6ff" : "#6b7b8d",
              border: "none",
              borderBottom:
                tab === t.id ? "2px solid #58a6ff" : "2px solid transparent",
              fontFamily: "'JetBrains Mono', 'DM Sans', monospace",
              fontSize: 12,
              fontWeight: tab === t.id ? 700 : 400,
              cursor: "pointer",
              transition: "color 0.2s",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <Suspense
        fallback={
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "80vh",
              color: "#484f58",
              background: "#010409",
              fontFamily: "monospace",
            }}
          >
            Loading...
          </div>
        }
      >
        <current.Component />
      </Suspense>
    </>
  );
}

createRoot(document.getElementById("root")).render(<Shell />);
