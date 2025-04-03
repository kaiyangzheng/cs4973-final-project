import { useRef, useState, useEffect } from "react";
import { Highlight } from "../components/pdf/types";
import { usePdfContext } from "./usePdfContext";
import { pdfjs } from "react-pdf";

export const usePdfViewer = () => {
  const {
    setSelectedText,
    setPdfContent,
    setPdfFileName,
    setPdfUrl,
    setSelectionInfo,
    clearPdf,
    extractAllText
  } = usePdfContext();
  
  const [file, setFile] = useState<File | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const pageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});
  const [isExtracting, setIsExtracting] = useState<boolean>(false);

  useEffect(() => {
    // Extract PDF text whenever a new file is uploaded
    const extractPdfContent = async () => {
      if (!file) return;
      
      try {
        setIsExtracting(true);
        
        // Create a URL for the file
        const fileUrl = URL.createObjectURL(file);
        setPdfUrl(fileUrl);
        setPdfFileName(file.name);
        
        // Load the PDF document
        const pdfDocument = await pdfjs.getDocument(fileUrl).promise;
        let fullText = "";
        
        // Extract text from each page
        for (let i = 1; i <= pdfDocument.numPages; i++) {
          const page = await pdfDocument.getPage(i);
          const textContent = await page.getTextContent();
          const pageText = textContent.items
            .map((item: any) => item.str)
            .join(" ");
          
          fullText += pageText + "\n\n--- Page Break ---\n\n";
        }
        
        // Convert the file to base64 for the server
        const reader = new FileReader();
        reader.onload = (e) => {
          const base64content = e.target?.result as string;
          // Remove the data URI prefix
          const base64Data = base64content.split(',')[1];
          setPdfContent(base64Data);
        };
        reader.readAsDataURL(file);
        
        // Store the full text of the PDF in localStorage for search
        localStorage.setItem('pdfText', fullText);
      } catch (error) {
        console.error("Error extracting PDF content:", error);
      } finally {
        setIsExtracting(false);
      }
    };
    
    extractPdfContent();
    
    // Clean up URLs when component unmounts
    return () => {
      if (file) {
        URL.revokeObjectURL(URL.createObjectURL(file));
      }
    };
  }, [file, setPdfContent, setPdfFileName, setPdfUrl]);

  const handleExtractAllText = async () => {
    if (!file) return;
    
    try {
      setIsExtracting(true);
      
      // Use the extractAllText function from PdfContext
      const success = await extractAllText();
      
      if (!success) {
        console.error("Failed to extract text from PDF");
        alert("Failed to extract text from the PDF. The file may be corrupted or password-protected.");
      }
      
    } catch (error) {
      console.error("Error extracting all PDF text:", error);
      alert("Failed to extract text from the PDF. The file may be corrupted or password-protected.");
    } finally {
      setIsExtracting(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setHighlights([]);
      clearPdf(); // Clear any previous PDF data
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  const handleTextSelection = (pageNumber: number) => {
    const selection = window.getSelection();

    if (!selection || selection.isCollapsed) return;

    const selectedText = selection.toString().trim();
    if (!selectedText) return;
    
    const range = selection.getRangeAt(0);
    const rects = Array.from(range.getClientRects());

    if (selectedText && rects.length > 0) {
      // Store the selection
      setHighlights((prev) => [
        ...prev,
        { text: selectedText, rects, pageNumber },
      ]);
      
      // Update the context with selected text
      setSelectedText(selectedText);
      
      // Store selection info
      const firstRect = rects[0];
      setSelectionInfo({
        pageNumber,
        position: { x: firstRect.left, y: firstRect.top },
        length: selectedText.length
      });
      
      console.log(`Selected text: "${selectedText}" on page ${pageNumber}`);
      
      // Clear the browser's selection
      // selection.removeAllRanges();
    }
  };

  const handleClearSelections = () => {
    setHighlights([]);
    setSelectedText(null);
    setSelectionInfo(null);
  };

  const handleRemoveFile = () => {
    setFile(null);
    setHighlights([]);
    pageRefs.current = {};
    clearPdf();
  };

  return {
    file,
    highlights,
    pageRefs,
    isExtracting,
    handleFileChange,
    handleTextSelection,
    handleClearSelections,
    handleRemoveFile,
    extractAllText: handleExtractAllText,
  };
};
