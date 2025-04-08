export interface Highlight {
  text: string;
  rects: DOMRect[];
  pageNumber: number;
}

export interface PdfContextType {
  file: File | null;
  allText: string;
  selectedText: string;
  highlights: Highlight[];
  pageRefs: React.RefObject<{ [key: number]: HTMLDivElement | null }>;
  handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleTextHighlight: (pageNumber: number) => void;
  handleClearSelections: () => void;
  handleRemoveFile: () => void;
  handleTextSelectionOnLoad: (items: unknown[]) => void;
  handleExtractAllText: () => void;
}
