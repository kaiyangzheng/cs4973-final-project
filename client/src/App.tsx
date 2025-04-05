import { BrowserRouter, Route, Routes } from "react-router";
import { Suspense } from "react";
import LoadingOrError from "./components/LoadingOrError";
import { PdfContextProvider } from "./hooks/PdfContext";
import React from "react";

const Home = React.lazy(() => import("./pages/Home"));

function App() {
  return (
    <>
      <PdfContextProvider>
        <BrowserRouter>
          <Suspense fallback={<LoadingOrError />}>
            <Routes>
              <Route path="/" element={<Home />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </PdfContextProvider>
    </>
  );
}

export default App;
