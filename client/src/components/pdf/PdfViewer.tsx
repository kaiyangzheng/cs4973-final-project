import { ReactElement, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Highlight } from "./types";
import "react-pdf/dist/esm/Page/TextLayer.css";
import { usePdfContext } from "../../hooks/usePdf";
import { FaFileAlt, FaTimes } from "react-icons/fa";

type PdfViewerProps = {
  file: File | null;
  highlights: Highlight[];
  pageRefs: React.RefObject<{ [key: number]: HTMLDivElement | null }>;
  handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleTextHighlight: (pageNumber: number) => void;
};

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function PdfViewer({
  file,
  highlights,
  pageRefs,
  handleFileChange,
  handleTextHighlight,
}: PdfViewerProps): ReactElement {
  const [numPages, setNumPages] = useState<number>(0);
  const {
    handleTextSelectionOnLoad,
    selectedText,
    handleRemoveFile,
    handleExtractAllText,
  } = usePdfContext();

  return (
    <div className="flex min-h-screen items-center justify-center">
      {!file ? (
        <label className="cursor-pointer rounded-lg bg-blue-500 px-4 py-2 text-white shadow-md transition hover:bg-blue-600">
          Upload PDF
          <input
            type="file"
            className="hidden"
            accept="application/pdf"
            onChange={handleFileChange}
          />
        </label>
      ) : (
        <div className="w-full max-w-2xl relative h-screen flex flex-col">
          <div className="flex justify-between items-center bg-gray-800 text-white p-2">
            <div className="flex items-center">
              <FaFileAlt className="mr-2" />
              <span className="text-sm truncate max-w-[200px]">
                {file.name}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleExtractAllText}
                className="bg-green-600 hover:bg-green-700 text-white text-xs px-2 py-1 rounded cursor-pointer"
                title="Extract all text from PDF"
              >
                Extract All Text
              </button>
              <button
                onClick={handleRemoveFile}
                className="bg-red-600 hover:bg-red-700 text-white rounded p-1 cursor-pointer"
                title="Remove PDF"
              >
                <FaTimes size={12} />
              </button>
            </div>
          </div>

          <div className="absolute top-8 left-0 right-0 z-10 bg-blue-100 text-blue-800 p-2 rounded border border-blue-300 m-2">
            <h3 className="font-bold">Selected text:</h3>
            <p className="text-sm">
              {selectedText?.substring(0, 200)}
              {selectedText && selectedText.length > 200 ? "..." : ""}
            </p>
            <p className="text-xs text-blue-600 mt-1">
              Ask questions about this specific selection!
            </p>
          </div>
          <div className="overflow-auto h-full p-3 rounded-lg shadow-md flex justify-center">
            <Document
              file={file}
              onLoadSuccess={({ numPages }) => setNumPages(numPages)}
              loading={
                <div className="flex items-center justify-center h-40">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
              }
              error={
                <div className="text-red-500 p-4 text-center">
                  <p className="font-bold">Error</p>
                  <p>Failed to load PDF. Please try another file.</p>
                </div>
              }
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
                    onMouseUp={() => handleTextHighlight(pageNumber)} // Detects selection per page
                  >
                    <Page
                      pageNumber={pageNumber}
                      renderTextLayer={true}
                      renderAnnotationLayer={false}
                      className="pdf-page"
                      loading={
                        <div className="h-80 w-full flex items-center justify-center">
                          <div className="animate-pulse bg-gray-200 h-full w-full"></div>
                        </div>
                      }
                      onGetTextSuccess={({ items }) => {
                        handleTextSelectionOnLoad(items);
                      }}
                    />

                    <div className="absolute top-0 right-0 bg-gray-800 text-white px-2 py-1 text-xs rounded-bl">
                      Page {pageNumber} of {numPages}
                    </div>

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
        </div>
      )}
    </div>
    // <div className="flex min-h-screen items-center justify-center">
    //   {!file ? (
    //     <label className="cursor-pointer rounded-lg bg-blue-500 px-4 py-2 text-white shadow-md transition hover:bg-blue-600">
    //       Upload File
    //       <input
    //         type="file"
    //         className="hidden"
    //         accept="application/pdf"
    //         onChange={handleFileChange}
    //       />
    //     </label>
    //   ) : (
    //     <div className="w-full max-w-2xl overflow-auto h-screen p-3 rounded-lg shadow-md flex justify-center">
    //       <Document file={file} onLoadSuccess={handleDocumentLoadSuccess}>
    //         {Array.from(new Array(numPages), (_, index) => {
    //           const pageNumber = index + 1;
    //           return (
    //             <div
    //               key={`page_${pageNumber}`}
    //               className="relative mb-4"
    //               ref={(el) => {
    //                 pageRefs.current[pageNumber] = el;
    //               }}
    //               onMouseUp={() => handleTextHighlight(pageNumber)} // Detects selection per page
    //             >
    //               <Page
    //                 pageNumber={pageNumber}
    //                 renderTextLayer={true}
    //                 renderAnnotationLayer={false}
    //                 onGetTextSuccess={({ items }) => {
    //                   handleTextSelectionOnLoad(items);
    //                 }}
    //               />

    //               {/* Render highlights for this page */}
    //               {highlights
    //                 .filter((h) => h.pageNumber === pageNumber)
    //                 .map((highlight, hIndex) =>
    //                   highlight.rects.map((rect, rIndex) => {
    //                     const pageRef = pageRefs.current[pageNumber];
    //                     if (!pageRef) return null;

    //                     const pageBounds = pageRef.getBoundingClientRect();
    //                     return (
    //                       <div
    //                         key={`${hIndex}-${rIndex}`}
    //                         className="absolute bg-yellow-300 opacity-50 pointer-events-none"
    //                         style={{
    //                           top: rect.top - pageBounds.top,
    //                           left: rect.left - pageBounds.left,
    //                           width: rect.width,
    //                           height: rect.height,
    //                         }}
    //                       />
    //                     );
    //                   })
    //                 )}
    //             </div>
    //           );
    //         })}
    //       </Document>
    //     </div>
    //   )}
    // </div>
  );
}
