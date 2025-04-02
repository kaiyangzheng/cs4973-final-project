import { useRef, useState } from "react";
import { Highlight } from "../components/pdf/types";

export const usePdfViewer = () => {
  const [file, setFile] = useState<File | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const pageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  const handleTextSelection = (pageNumber: number) => {
    const selection = window.getSelection();

    if (!selection || selection.isCollapsed) return;

    const selectedText = selection.toString();
    const range = selection.getRangeAt(0);
    const rects = Array.from(range.getClientRects());

    if (selectedText.trim() && rects.length > 0) {
      setHighlights((prev) => [
        ...prev,
        { text: selectedText, rects, pageNumber },
      ]);
      selection.removeAllRanges();
    }
  };

  const handleClearSelections = () => {
    setHighlights([]);
  };

  const handleRemoveFile = () => {
    setFile(null);
    setHighlights([]);
    pageRefs.current = {};
  };

  return {
    file,
    highlights,
    pageRefs,
    handleFileChange,
    handleTextSelection,
    handleClearSelections,
    handleRemoveFile,
  };
};
