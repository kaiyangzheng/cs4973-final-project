import { ReactElement, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Highlight } from "./types";
import "react-pdf/dist/esm/Page/TextLayer.css";

type PdfViewerProps = {
  file: File | null;
  highlights: Highlight[];
  pageRefs: React.RefObject<{ [key: number]: HTMLDivElement | null }>;
  handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleTextSelection: (pageNumber: number) => void;
};

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function PdfViewer({
  file,
  highlights,
  pageRefs,
  handleFileChange,
  handleTextSelection,
}: PdfViewerProps): ReactElement {
  const [numPages, setNumPages] = useState<number>(0);

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
        <div className="w-full max-w-2xl overflow-auto h-screen p-3 rounded-lg shadow-md flex justify-center">
          <Document
            file={file}
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          >
            {Array.from(new Array(numPages), (_, index) => {
              const pageNumber = index + 1;
              return (
                <div
                  key={`page_${pageNumber}`}
                  className="relative mb-4"
                  ref={(el) => {
                    pageRefs.current[pageNumber] = el;
                  }}
                  onMouseUp={() => handleTextSelection(pageNumber)} // Detects selection per page
                >
                  <Page
                    pageNumber={pageNumber}
                    renderTextLayer={true}
                    renderAnnotationLayer={false}
                  />

                  {/* Render highlights for this page */}
                  {highlights
                    .filter((h) => h.pageNumber === pageNumber)
                    .map((highlight, hIndex) =>
                      highlight.rects.map((rect, rIndex) => {
                        const pageRef = pageRefs.current[pageNumber];
                        if (!pageRef) return null;

                        const pageBounds = pageRef.getBoundingClientRect();
                        return (
                          <div
                            key={`${hIndex}-${rIndex}`}
                            className="absolute bg-yellow-300 opacity-50 pointer-events-none"
                            style={{
                              top: rect.top - pageBounds.top,
                              left: rect.left - pageBounds.left,
                              width: rect.width,
                              height: rect.height,
                            }}
                          />
                        );
                      })
                    )}
                </div>
              );
            })}
          </Document>
        </div>
      )}
    </div>
  );
}
