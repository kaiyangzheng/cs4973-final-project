import React, { createContext, useContext, useState, ReactNode, FC } from 'react';
import { pdfjs } from 'react-pdf';

interface PdfContextType {
  pdfContent: any | null;
  setPdfContent: (content: any | null) => void;
  selectedText: string | null;
  setSelectedText: (text: string | null) => void;
  isSelectionActive: boolean;
  clearSelection: () => void;
  pdfFileName: string | null;
  selectionInfo: { pageNumber: number; length: number } | null;
  extractAllText: () => Promise<string | null>;
  setSelectionInfo: (info: { pageNumber: number; length: number } | null) => void;
}

const PdfContext = createContext<PdfContextType | undefined>(undefined);

interface PdfProviderProps {
  children: ReactNode;
}

export const PdfProvider: FC<PdfProviderProps> = ({ children }) => {
  const [pdfContent, setPdfContent] = useState<any | null>(null);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [pdfFileName, setPdfFileName] = useState<string | null>(null);
  const [selectionInfo, setSelectionInfo] = useState<{ pageNumber: number; length: number } | null>(null);

  const isSelectionActive = Boolean(selectedText && selectedText.trim().length > 0);

  const clearSelection = () => {
    setSelectedText(null);
    setSelectionInfo(null);
  };

  const extractAllText = async (): Promise<string | null> => {
    if (!pdfContent) return null;
    try {
      // More detailed logging to understand the content type
      console.log("PDF content type:", typeof pdfContent);
      console.log("PDF content constructor:", pdfContent.constructor ? pdfContent.constructor.name : "unknown");
      
      // Handle direct binary content
      if (typeof pdfContent === 'string' && pdfContent.startsWith('%PDF')) {
        console.error("Raw PDF binary data detected. Cannot extract text.");
        return null;
      }
      
      // Handle ArrayBuffer content by using PDF.js directly
      if (pdfContent instanceof ArrayBuffer || 
          (typeof pdfContent === 'object' && pdfContent.byteLength)) {
        console.log("Detected ArrayBuffer content, loading with PDF.js...");
        try {
          // Use pdfjs from react-pdf instead of dynamic import
          const loadingTask = pdfjs.getDocument({data: pdfContent});
          const pdf = await loadingTask.promise;
          
          console.log(`PDF loaded successfully. Number of pages: ${pdf.numPages}`);
          
          // Extract text from all pages
          let fullText = '';
          for (let i = 1; i <= pdf.numPages; i++) {
            try {
              const page = await pdf.getPage(i);
              const textContent = await page.getTextContent();
              const pageText = textContent.items
                .map((item: any) => item.str)
                .join(' ');
              fullText += `--- Page ${i} ---\n${pageText}\n\n`;
            } catch (pageError) {
              console.error(`Error extracting text from page ${i}:`, pageError);
            }
          }
          return fullText.trim() || null;
        } catch (pdfError) {
          console.error("Error processing PDF with PDF.js:", pdfError);
          return null;
        }
      }
      
      // Handle PDF.js document
      if (pdfContent._pdfInfo) {
        console.log("Detected PDF.js Document object");
        let fullText = '';
        const numPages = pdfContent._pdfInfo.numPages;
        console.log(`Extracting text from ${numPages} pages...`);
        
        for (let i = 1; i <= numPages; i++) {
          try {
            const page = await pdfContent.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items
              .map((item: any) => item.str)
              .join(' ');
            fullText += `--- Page ${i} ---\n${pageText}\n\n`;
          } catch (pageError) {
            console.error(`Error extracting text from page ${i}:`, pageError);
          }
        }
        return fullText.trim() || null;
      }
      
      // Use pdfContent.getText if available
      if (typeof pdfContent.getText === 'function') {
        console.log("Using getText() method");
        const text = await pdfContent.getText();
        return text || null;
      }
      
      console.error("Unknown PDF content format, cannot extract text:", pdfContent);
      return null;
    } catch (error) {
      console.error("Error extracting text:", error);
      return null;
    }
  };

  const value = {
    pdfContent,
    setPdfContent,
    selectedText,
    setSelectedText,
    isSelectionActive,
    clearSelection,
    pdfFileName,
    selectionInfo,
    extractAllText,
    setSelectionInfo
  };

  return React.createElement(PdfContext.Provider, { value }, children);
};

export const usePdfContext = () => {
  const context = useContext(PdfContext);
  if (context === undefined) {
    throw new Error('usePdfContext must be used within a PdfProvider');
  }
  return context;
}; 