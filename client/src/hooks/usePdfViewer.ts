import { useState, useRef, useCallback } from "react";
import { Highlight } from "../components/pdf/types";
import { usePdfContext } from "./usePdfContext";

export function usePdfViewer() {
  const [file, setFile] = useState<File | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  const pageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});
  const { setSelectedText, setPdfContent, pdfContent, extractAllText: contextExtractAllText } = usePdfContext();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const selectedFile = files[0];
      
      // Verify it's a PDF
      if (!selectedFile.type.includes('pdf')) {
        alert('Please select a PDF file.');
        return;
      }
      
      setFile(selectedFile);
      
      // Read the file as ArrayBuffer for PDF.js to process
      // Do NOT read as text which leads to binary extraction issues
      const reader = new FileReader();
      reader.onload = async (e) => {
        if (e.target?.result) {
          // Set the PDF content in the context
          // This will be handled by PDF.js in the PdfViewer component
          setPdfContent(e.target.result);
        }
      };
      reader.readAsArrayBuffer(selectedFile);
    }
  };

  const extractPdfText = async (): Promise<string | null> => {
    setIsExtracting(true);
    try {
      if (!pdfContent) {
        console.error("No PDF content available for extraction");
        return null;
      }

      // Use the context's extractAllText function
      const extractedText = await contextExtractAllText();
      
      if (extractedText) {
        // If extraction was successful, set the extracted text as selected
        setSelectedText(extractedText);
        return extractedText;
      }
      
      return null;
    } catch (error) {
      console.error("Error extracting PDF text:", error);
      return null;
    } finally {
      setIsExtracting(false);
    }
  };

  const handleTextSelection = useCallback(
    (pageNumber: number) => {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        return;
      }

      // Get the selected text
      const selectedText = selection.toString().trim();
      if (!selectedText) return;

      // Set the selected text in the context
      setSelectedText(selectedText);
      
      // Create highlight rectangles
      const range = selection.getRangeAt(0);
      const rects = Array.from(range.getClientRects());

      // Convert client coordinates to page coordinates
      const pageRef = pageRefs.current[pageNumber];
      if (!pageRef) return;

      const pageBounds = pageRef.getBoundingClientRect();
      const highlight: Highlight = {
        pageNumber,
        text: selectedText,
        rects: rects.map((rect) => ({
          top: rect.top,
          left: rect.left,
          width: rect.width,
          height: rect.height,
        })),
      };

      setHighlights((prev) => [...prev, highlight]);
    },
    [setSelectedText]
  );

  const handleClearSelections = () => {
    setHighlights([]);
    setSelectedText(null);
  };

  const handleRemoveFile = () => {
    setFile(null);
    setHighlights([]);
    setSelectedText(null);
    setPdfContent(null);
  };

  return {
    file,
    highlights,
    pageRefs,
    handleFileChange,
    handleTextSelection,
    handleClearSelections,
    handleRemoveFile,
    extractAllText: extractPdfText,
    isExtracting
  };
} 