import { Skeleton } from "../components/ui";

/** Placeholder with the page's own heading, so the layout does not jump when data arrives. */
export function ProgressLoading() {
  return (
    <section aria-busy="true">
      <div className="page-head">
        <h1>Progress</h1>
        <p className="muted" role="status">Loading your progress…</p>
      </div>
      <div className="stat-grid">
        <Skeleton className="skeleton-tile" />
        <Skeleton className="skeleton-tile" />
        <Skeleton className="skeleton-tile" />
        <Skeleton className="skeleton-tile" />
      </div>
      <h2 className="section-title"><span className="sr-only">Charts</span></h2>
      <Skeleton className="skeleton-chart" />
    </section>
  );
}
