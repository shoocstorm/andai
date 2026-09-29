// Workspace layout preferences, persisted across launches.
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

type LayoutState = {
  /** Left nav shows icons only. */
  navCollapsed: boolean;
  /** Execution Trace panel visible in Command Center (hidden until the user opens it). */
  traceOpen: boolean;
  toggleNav: () => void;
  toggleTrace: () => void;
};

export const useLayout = create<LayoutState>()(
  persist(
    (set) => ({
      navCollapsed: false,
      traceOpen: false,
      toggleNav: () => set((s) => ({ navCollapsed: !s.navCollapsed })),
      toggleTrace: () => set((s) => ({ traceOpen: !s.traceOpen })),
    }),
    {
      name: 'andai.layout',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ navCollapsed: s.navCollapsed, traceOpen: s.traceOpen }),
    },
  ),
);
