import { ReactElement, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

// TODO: Implement text highlighting (https://chatgpt.com/c/67eb6e53-c120-8012-8fbd-90bc9a0740d5 - this code kind of works)
export default function PdfViewer(): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [numPages, setNumPages] = useState<number>(0);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center">
      {!file ? (
        <label className="cursor-pointer rounded-lg bg-blue-500 px-4 py-2 text-white shadow-md transition hover:bg-blue-600">
          Upload File
          <input
            type="file"
            className="hidden"
            accept="application/pdf"
            onChange={handleFileChange}
          />
        </label>
      ) : (
        <div className="w-full max-w-2xl overflow-auto h-screen p-3 border border-gray-300 rounded-lg shadow-md flex justify-center">
          <Document
            file={file}
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          >
            {Array.from(new Array(numPages), (_, index) => (
              <Page
                key={`page_${index + 1}`}
                pageNumber={index + 1}
                renderTextLayer={true} // Enables text selection
                renderAnnotationLayer={false}
                className="mb-4"
              />
            ))}
          </Document>
        </div>
      )}
    </div>
  );
}
