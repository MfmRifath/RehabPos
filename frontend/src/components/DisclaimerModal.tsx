import { ShieldCheck } from "lucide-react";
import { DISCLAIMER } from "../lib/disclaimer";
import { BrandMark } from "./ui";
import { Modal } from "./Modal";

export function DisclaimerModal({ onAccept }: { onAccept: () => void }) {
  return (
    <Modal labelledBy="disclaimer-title">
      <span className="modal-mark"><BrandMark size={36} /></span>
      <h2 id="disclaimer-title">Before you start</h2>
      <p className="disclaimer" data-testid="disclaimer-text">{DISCLAIMER}</p>
      <p className="note">
        <ShieldCheck size={20} aria-hidden="true" />
        <span>Video stays in your browser. Only body landmarks are sent to the local RehabPose server.</span>
      </p>
      <button className="btn btn-primary btn-lg btn-block" onClick={onAccept} autoFocus>
        I understand
      </button>
    </Modal>
  );
}
