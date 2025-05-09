import { ReactElement } from "react";
import Head from "../components/Head";
import Chat from "../components/chat/Chat";
import PdfViewer from "../components/pdf/PdfViewer";
import { usePdfContext } from "../hooks/usePdf";

export default function Home(): ReactElement {
  const {
    file,
    highlights,
    pageRefs,
    handleFileChange,
    handleTextHighlight,
    handleClearSelections,
    handleRemoveFile,
  } = usePdfContext();

  return (
    <>
      <Head title="Chat" />
      <div className="flex min-h-screen flex-row items-center justify-center">
        {file && (
          <div className="absolute top-4 left-4 flex flex-col gap-3 items-start">
            <button
              onClick={handleClearSelections}
              className="hover:cursor-pointer p-2 rounded-lg bg-blue-500 text-white shadow-md transition hover:bg-blue-600 w-full"
            >
              Clear Selections
            </button>

            <button
              onClick={handleRemoveFile}
              className="hover:cursor-pointer p-2 rounded-lg bg-red-500 text-white shadow-md transition hover:bg-red-600 w-full"
            >
              Remove File
            </button>
          </div>
        )}

        <div className="flex-[0.65]">
          <PdfViewer
            file={file}
            highlights={highlights}
            pageRefs={pageRefs}
            handleFileChange={handleFileChange}
            handleTextHighlight={handleTextHighlight}
          />
        </div>
        {file && (
          <div className="flex-[0.35]">
            <Chat />
          </div>
        )}
      </div>
    </>
  );
}
