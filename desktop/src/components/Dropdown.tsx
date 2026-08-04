import { useEffect, useId, useRef, useState } from "react";
import { CheckIcon, ChevronIcon } from "../icons";

export interface DropdownOption {
  value: string;
  label: string;
  /** Right-aligned secondary text, e.g. a model's size on disk. */
  hint?: string;
  /** Dot colour in front of the label; omit for options with no state. */
  status?: "ready" | "missing";
}

/** Listbox dropdown replacing the native `<select>`.
 *
 * A native select cannot show per-option state (downloaded vs not, size), and
 * on macOS it ignores the app's own palette. This renders the same semantics
 * (`combobox` + `listbox` roles, arrow/Home/End/Enter/Escape keys) with the
 * app's tokens.
 */
export default function Dropdown({
  value,
  options,
  onChange,
  label,
  placeholder = "None",
}: {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const listId = useId();

  const selectedIndex = options.findIndex((option) => option.value === value);
  const selected = selectedIndex >= 0 ? options[selectedIndex] : undefined;

  useEffect(() => {
    if (!open) return;
    setActive(selectedIndex >= 0 ? selectedIndex : 0);
    const onPointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open, selectedIndex]);

  useEffect(() => {
    if (!open) return;
    listRef.current
      ?.querySelector<HTMLLIElement>(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const commit = (index: number) => {
    const option = options[index];
    if (option) onChange(option.value);
    setOpen(false);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!options.length) return;
    switch (event.key) {
      case "ArrowDown":
      case "ArrowUp": {
        event.preventDefault();
        const step = event.key === "ArrowDown" ? 1 : -1;
        if (!open) {
          setOpen(true);
          return;
        }
        setActive((prev) => (prev + step + options.length) % options.length);
        break;
      }
      case "Home":
        if (open) {
          event.preventDefault();
          setActive(0);
        }
        break;
      case "End":
        if (open) {
          event.preventDefault();
          setActive(options.length - 1);
        }
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        if (open) commit(active);
        else setOpen(true);
        break;
      case "Escape":
        if (open) {
          event.preventDefault();
          setOpen(false);
        }
        break;
    }
  };

  return (
    <div className="dropdown" ref={rootRef}>
      <button
        type="button"
        className={open ? "dropdown-trigger open" : "dropdown-trigger"}
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-haspopup="listbox"
        aria-label={label}
        onClick={() => setOpen((prev) => !prev)}
        onKeyDown={onKeyDown}
      >
        <span className="dropdown-value">
          {selected ? (
            <>
              {selected.status && <span className={`dot ${selected.status}`} />}
              {selected.label}
            </>
          ) : (
            <span className="dropdown-placeholder">{placeholder}</span>
          )}
        </span>
        <ChevronIcon />
      </button>
      {open && (
        <ul className="dropdown-menu" role="listbox" id={listId} ref={listRef}>
          {options.length === 0 && <li className="dropdown-empty">No models available</li>}
          {options.map((option, index) => (
            <li
              key={option.value}
              data-index={index}
              role="option"
              aria-selected={option.value === value}
              className={index === active ? "dropdown-option active" : "dropdown-option"}
              onMouseEnter={() => setActive(index)}
              onClick={() => commit(index)}
            >
              <span className="dropdown-check">
                {option.value === value && <CheckIcon />}
              </span>
              {option.status && <span className={`dot ${option.status}`} />}
              <span className="dropdown-label">{option.label}</span>
              {option.hint && <span className="dropdown-hint">{option.hint}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
