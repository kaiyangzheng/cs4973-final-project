import { BrowserRouter, Route, Routes } from "react-router";
import { Suspense } from "react";
import LoadingOrError from "./components/LoadingOrError";
import React from "react";
import { PdfProvider } from "./hooks/usePdfContext";

const Home = React.lazy(() => import("./pages/Home"));

function App() {
  return (
    <>
      <PdfProvider>
        <BrowserRouter>
          <Suspense fallback={<LoadingOrError />}>
            <Routes>
              <Route path="/" element={<Home />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </PdfProvider>
    </>
  );
}

export default App;
