import { createContext, useContext } from "react";
import { PdfContextType } from "./types";

export const PdfContext = createContext<PdfContextType>({} as PdfContextType);
export const usePdfContext = () => useContext(PdfContext);
