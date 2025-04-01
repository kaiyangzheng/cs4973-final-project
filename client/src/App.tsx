import { BrowserRouter, Route, Routes } from "react-router";
import { Suspense } from "react";
import LoadingOrError from "./components/LoadingOrError";
import React from "react";

const Home = React.lazy(() => import("./pages/Home"));

function App() {
  return (
    <>
      <BrowserRouter>
        <Suspense fallback={<LoadingOrError />}>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </>
  );
}

export default App;
