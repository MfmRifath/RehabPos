import { useEffect, useRef, type KeyboardEvent, type ReactNode } from "react";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Modal dialog with a focus trap; focus returns to the previous element on close.
 * Without `onDismiss` it cannot be closed with Esc (the disclaimer must be acknowledged).
 */
export function Modal({
  labelledBy,
  onDismiss,
  role = "dialog",
  children,
}: {
  labelledBy: string;
  onDismiss?: () => void;
  role?: "dialog" | "alertdialog";
  children: ReactNode;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const node = panel.current;
    if (node && !node.contains(document.activeElement)) node.querySelector<HTMLElement>(FOCUSABLE)?.focus();
    return () => previous?.focus?.();
  }, []);

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    e.stopPropagation(); // Page shortcuts must not fire behind the dialog.
    if (e.key === "Escape") {
      onDismiss?.();
      return;
    }
    if (e.key !== "Tab" || !panel.current) return;
    const items = [...panel.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
    const first = items[0];
    const last = items.at(-1);
    if (!first || !last) return;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="modal-backdrop" onKeyDown={onKeyDown}>
      <div ref={panel} className="modal" role={role} aria-modal="true" aria-labelledby={labelledBy} data-testid="modal">
        {children}
      </div>
    </div>
  );
}
