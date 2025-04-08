import React, { useRef, useState } from "react";
import { Highlight } from "./types";
import { PdfContext } from "./usePdf";

export const PdfContextProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [selectedText, setSelectedText] = useState<string>("");
  const [allText, setAllText] = useState<string>("");
  const pageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  const handleTextHighlight = (pageNumber: number) => {
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
      setSelectedText(selectedText);
      selection.removeAllRanges();
    }
  };

  const handleTextSelectionOnLoad = (items: unknown[]) => {
    const text = items.map((item) => (item as { str: string }).str).join(" ");
    setSelectedText((prev) => {
      if (!prev) return text;
      return `${prev}\n${text}`;
    });
    setAllText((prev) => {
      if (!prev) return text;
      return `${prev}\n${text}`;
    });
  };

  const handleExtractAllText = () => {
    setSelectedText(allText);
    setHighlights([]);
  };

  const handleClearSelections = () => {
    setHighlights([]);
    setSelectedText("");
  };

  const handleRemoveFile = () => {
    setFile(null);
    setHighlights([]);
    setSelectedText("");
    pageRefs.current = {};
  };

  return (
    <PdfContext.Provider
      value={{
        file,
        allText,
        selectedText,
        highlights,
        pageRefs,
        handleFileChange,
        handleTextHighlight,
        handleTextSelectionOnLoad,
        handleExtractAllText,
        handleClearSelections,
        handleRemoveFile,
      }}
    >
      {children}
    </PdfContext.Provider>
  );
};
