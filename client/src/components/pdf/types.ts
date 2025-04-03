export interface RectangleArea {
    top: number;
    left: number;
    width: number;
    height: number;
  }
  
  export interface Highlight {
    pageNumber: number;
    text: string;
    rects: RectangleArea[];
  }