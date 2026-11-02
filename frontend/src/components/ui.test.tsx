import { fireEvent, render, screen } from "@testing-library/react";
import { Modal } from "./Modal";
import { RepCounter } from "./RepCounter";
import { RomGauge } from "./RomGauge";
import { Segmented, Stepper } from "./ui";

const dialog = (onDismiss?: () => void) => (
  <Modal labelledBy="t" onDismiss={onDismiss}>
    <h2 id="t">Title</h2>
    <button>First</button>
    <button>Last</button>
  </Modal>
);

describe("Modal", () => {
  it("moves focus inside on open and returns it on close", () => {
    const opener = document.createElement("button");
    document.body.append(opener);
    opener.focus();
    const { unmount } = render(dialog());
    expect(screen.getByRole("button", { name: "First" })).toHaveFocus();
    unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it("wraps Tab in both directions", () => {
    render(dialog());
    const first = screen.getByRole("button", { name: "First" });
    const last = screen.getByRole("button", { name: "Last" });
    fireEvent.keyDown(first, { key: "Tab", shiftKey: true });
    expect(last).toHaveFocus();
    fireEvent.keyDown(last, { key: "Tab" });
    expect(first).toHaveFocus();
  });

  it("closes on Esc only when it can be dismissed", () => {
    const onDismiss = vi.fn();
    const { unmount } = render(dialog(onDismiss));
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onDismiss).toHaveBeenCalledTimes(1);
    unmount();
    render(dialog());
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("keeps page shortcuts from firing behind the dialog", () => {
    const onKey = vi.fn();
    window.addEventListener("keydown", onKey);
    render(dialog());
    fireEvent.keyDown(screen.getByRole("button", { name: "First" }), { key: "m" });
    expect(onKey).not.toHaveBeenCalled();
    window.removeEventListener("keydown", onKey);
  });
});

describe("RomGauge", () => {
  const dash = () => screen.getByTestId("rom-gauge-fill").getAttribute("stroke-dasharray");

  it("fills in proportion to the target", () => {
    render(<RomGauge value={75} target={150} unit="deg" />);
    expect(dash()).toBe("50.0 100");
    expect(screen.getByRole("img")).toHaveAccessibleName("Current 75°, target 150°");
  });

  it("stops at the target rather than rewarding more range", () => {
    render(<RomGauge value={200} target={150} unit="deg" />);
    expect(dash()).toBe("100.0 100");
    expect(screen.getByText("200°")).toBeInTheDocument();
  });

  it("shows no fill and a dash with no reading yet", () => {
    render(<RomGauge value={null} target={150} unit="deg" />);
    expect(dash()).toBe("0.0 100");
    expect(screen.getByText("–")).toBeInTheDocument();
  });

  it("keeps two decimals for ratio-unit exercises", () => {
    render(<RomGauge value={0.76} target={0.9} unit="ratio" />);
    expect(screen.getByText("0.76")).toBeInTheDocument();
    expect(screen.getByText("target 0.90")).toBeInTheDocument();
  });
});

describe("RepCounter", () => {
  it("announces only the rep count, not the points beside it", () => {
    const { container } = render(<RepCounter count={3} points={30} combo={2} gain={10} gainKey={3} />);
    const live = container.querySelectorAll("[aria-live]");
    expect(live).toHaveLength(1);
    expect(live[0]).toBe(screen.getByTestId("rep-count"));
    expect(live[0]).toHaveTextContent(/^3$/);
    expect(screen.getByText("30 pts · combo 2")).toBeInTheDocument();
  });

  it("hides the gain until a rep has earned points", () => {
    render(<RepCounter count={0} points={0} combo={0} gain={0} gainKey={0} />);
    expect(screen.queryByText("+0")).toBeNull();
  });
});

it("stepper marks only the current step", () => {
  render(<Stepper current={1} />);
  const current = screen.getAllByRole("listitem").filter((li) => li.getAttribute("aria-current") === "step");
  expect(current).toHaveLength(1);
  expect(current[0]).toHaveTextContent("Camera check");
});

it("segmented control is a labelled radio group", () => {
  const onChange = vi.fn();
  render(<Segmented legend="Which arm?" name="side" value="R" options={[{ value: "R", label: "Right arm" }, { value: "L", label: "Left arm" }]} onChange={onChange} />);
  expect(screen.getByRole("group", { name: "Which arm?" })).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: "Right arm" })).toBeChecked();
  fireEvent.click(screen.getByRole("radio", { name: "Left arm" }));
  expect(onChange).toHaveBeenCalledWith("L");
});
