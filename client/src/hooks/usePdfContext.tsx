import React, { createContext, useContext, useState, ReactNode } from "react";
import { pdfjs } from "react-pdf";

// Define the type for the PDF context
export interface PdfContextType {
  pdfContent: string | null;
  setPdfContent: (content: string | null) => void;
  pdfFileName: string | null;
  setPdfFileName: (name: string | null) => void;
  pdfUrl: string | null;
  setPdfUrl: (url: string | null) => void;
  selectedText: string | null;
  setSelectedText: (text: string | null) => void;
  isSelectionActive: boolean;
  clearSelection: () => void;
  clearPdf: () => void;
  selectionInfo: {
    pageNumber: number;
    position?: { x: number; y: number };
    length: number;
  } | null;
  setSelectionInfo: (info: {
    pageNumber: number;
    position?: { x: number; y: number };
    length: number;
  } | null) => void;
  extractAllText: () => Promise<boolean>;
}

// Create the PDF context with default values
const PdfContext = createContext<PdfContextType>({
  pdfContent: null,
  setPdfContent: () => {},
  pdfFileName: null,
  setPdfFileName: () => {},
  pdfUrl: null,
  setPdfUrl: () => {},
  selectedText: null,
  setSelectedText: () => {},
  isSelectionActive: false,
  clearSelection: () => {},
  clearPdf: () => {},
  selectionInfo: null,
  setSelectionInfo: () => {},
  extractAllText: async () => false,
});

// Provider component
export const PdfProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [pdfContent, setPdfContent] = useState<string | null>(null);
  const [pdfFileName, setPdfFileName] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [selectionInfo, setSelectionInfo] = useState<{
    pageNumber: number;
    position?: { x: number; y: number };
    length: number;
  } | null>(null);

  // Calculate isSelectionActive based on whether selectedText exists
  const isSelectionActive = !!selectedText;

  // Clear the selected text
  const clearSelection = () => {
    setSelectedText(null);
    setSelectionInfo(null);
  };

  // Clear all PDF related data
  const clearPdf = () => {
    setPdfContent(null);
    setPdfFileName(null);
    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl);
    }
    setPdfUrl(null);
    clearSelection();
  };

  // Extract all text from the PDF using PDF.js
  const extractAllText = async (): Promise<boolean> => {
    if (!pdfUrl) return false;
    
    try {
      // Load the PDF document
      const pdfDocument = await pdfjs.getDocument({ url: pdfUrl, disableAutoFetch: false, disableStream: false }).promise;
      let fullText = "";
      let extractedPageCount = 0;
      const totalPages = pdfDocument.numPages;
      console.log(`Extracting text from ${totalPages} pages...`);

      // First try with PDF.js text content
      for (let i = 1; i <= totalPages; i++) {
        const page = await pdfDocument.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items
          .map((item: any) => item.str)
          .join(" ")
          .trim();

        if (pageText) {
          fullText += `\n\nPage ${i} of ${totalPages}:\n${pageText}`;
          extractedPageCount++;
        }
      }

      // If we got some text, use it
      if (fullText.trim()) {
        console.log(`Successfully extracted text from ${extractedPageCount} pages`);
        setSelectedText(fullText.trim());
        setSelectionInfo({
          pageNumber: 0,
          length: fullText.length
        });
        return true;
      }

      // If PDF.js extraction failed, try alternative methods
      console.log("PDF.js extraction failed, trying alternative methods...");
      
      // Try with canvas
      for (let i = 1; i <= totalPages; i++) {
        const page = await pdfDocument.getPage(i);
        const viewport = page.getViewport({ scale: 1.0 });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) continue;

        canvas.height = viewport.height;
        canvas.width = viewport.width;

        await page.render({
          canvasContext: context,
          viewport: viewport
        }).promise;

        // Try to extract text from canvas
        const canvasText = canvas.textContent || "";
        if (canvasText) {
          fullText += `\n\nPage ${i} of ${totalPages}:\n${canvasText}`;
          extractedPageCount++;
        }
      }

      if (fullText.trim()) {
        console.log(`Successfully extracted text from ${extractedPageCount} pages using canvas`);
        setSelectedText(fullText.trim());
        setSelectionInfo({
          pageNumber: 0,
          length: fullText.length
        });
        return true;
      }

      // If all else fails, try DOM-based extraction
      console.log("Canvas extraction failed, trying DOM-based extraction...");
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(document.body);
        selection.removeAllRanges();
        selection.addRange(range);
        
        const extractedText = selection.toString();
        selection.removeAllRanges();
        
        if (extractedText && extractedText.length > 200) {
          console.log("Successfully extracted text using DOM selection");
          setSelectedText(extractedText);
          setSelectionInfo({
            pageNumber: 0,
            length: extractedText.length
          });
          return true;
        }
      }

      console.log("All extraction methods failed");
      return false;

    } catch (error) {
      console.error("Error extracting text:", error);
      return false;
    }
  };
  
  return (
    <PdfContext.Provider
      value={{
        pdfContent,
        setPdfContent,
        pdfFileName,
        setPdfFileName,
        pdfUrl,
        setPdfUrl,
        selectedText,
        setSelectedText,
        isSelectionActive,
        clearSelection,
        clearPdf,
        selectionInfo,
        setSelectionInfo,
        extractAllText,
      }}
    >
      {children}
    </PdfContext.Provider>
  );
}; 

// Custom hook to use the PDF context
export const usePdfContext = () => useContext(PdfContext); 