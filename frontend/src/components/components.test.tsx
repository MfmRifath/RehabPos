import { fireEvent, render, screen } from "@testing-library/react";
import { DISCLAIMER } from "../lib/disclaimer";
import { CueBanner } from "./CueBanner";
import { DisclaimerModal } from "./DisclaimerModal";

it("cue banner shows text with severity class and a non-colour label", () => {
  const { rerender } = render(<CueBanner cue={{ code: "TRUNK_LEAN", text: "Keep your trunk upright", speak: true, severity: "warn" }} />);
  const el = screen.getByTestId("cue");
  expect(el).toHaveClass("warn");
  expect(el).toHaveTextContent("Check form:");
  expect(el).toHaveTextContent("Keep your trunk upright");
  rerender(<CueBanner cue={null} />);
  expect(screen.queryByTestId("cue")).toBeNull();
});

it("disclaimer modal shows the verbatim text and accepts", () => {
  const onAccept = vi.fn();
  render(<DisclaimerModal onAccept={onAccept} />);
  expect(screen.getByTestId("disclaimer-text")).toHaveTextContent(DISCLAIMER);
  fireEvent.click(screen.getByRole("button", { name: /understand/i }));
  expect(onAccept).toHaveBeenCalled();
});
