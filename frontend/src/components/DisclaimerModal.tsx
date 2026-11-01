import { DISCLAIMER } from "../lib/disclaimer";

export function DisclaimerModal({ onAccept }: { onAccept: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="disclaimer-title">
      <div className="card modal">
        <h2 id="disclaimer-title">Before you start</h2>
        <p className="disclaimer" data-testid="disclaimer-text">{DISCLAIMER}</p>
        <p className="muted">Video stays in your browser. Only body landmarks are sent to the local RehabPose server.</p>
        <button className="primary" onClick={onAccept} autoFocus>
          I understand
        </button>
      </div>
    </div>
  );
}
