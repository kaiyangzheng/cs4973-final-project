import { usePdfViewer } from "../../hooks/usePdfViewer";
import PdfViewer from "./PdfViewer";

export default function PdfPage() {
  const {
    file,
    highlights,
    pageRefs,
    isExtracting,
    handleFileChange,
    handleTextSelection,
    handleClearSelections,
    handleRemoveFile,
    extractAllText,
  } = usePdfViewer();

  return (
    <div>
      <PdfViewer
        file={file}
        highlights={highlights}
        pageRefs={pageRefs}
        handleFileChange={handleFileChange}
        handleTextSelection={handleTextSelection}
        isExtracting={isExtracting}
        extractAllText={extractAllText}
        handleRemoveFile={handleRemoveFile}
      />
    </div>
  );
} 