import { ReactElement } from "react";
import Head from "../components/Head";
import Chat from "../components/chat/Chat";
import PdfViewer from "../components/pdf/PdfViewer";

export default function Home(): ReactElement {
  return (
    <>
      <Head title="Chat" />
      <div className="flex min-h-screen flex-row items-center justify-center">
        <div className="flex-[0.7]">
          <PdfViewer />
        </div>
        <div className="flex-[0.3]">
          <Chat />
        </div>
      </div>
    </>
  );
}
