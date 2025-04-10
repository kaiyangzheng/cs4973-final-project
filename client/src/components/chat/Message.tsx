import { ReactElement, useEffect } from "react";
import { MessageType } from "./types";
import { FaPaintBrush, FaFile, FaSpinner, FaTag } from "react-icons/fa";
import Markdown from "react-markdown";

interface CategoryObject {
  code: string;
  label: string;
}

export default function Message({
  message,
}: {
  message: MessageType;
}): ReactElement {
  // More detailed debug logging to help troubleshoot
  useEffect(() => {
    if (message.metadata?.paper_categories) {
      console.log("Paper categories received:", JSON.stringify(message.metadata.paper_categories, null, 2));
      console.log("Type of first category:", message.metadata.paper_categories.length > 0 ? 
        typeof message.metadata.paper_categories[0] : "none");
      if (message.metadata.paper_categories.length > 0 && typeof message.metadata.paper_categories[0] === 'object') {
        console.log("Properties:", Object.keys(message.metadata.paper_categories[0]));
      }
    }
  }, [message.metadata?.paper_categories]);

  // System messages should be centered and have a distinctive style
  if (message.role === "system") {
    return (
      <div className="flex justify-center my-2 px-2">
        <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-2 rounded">
          {message.content}
        </div>
      </div>
    );
  }

  // Check if this is an extraction in progress message
  const isExtractionInProgress =
    message.metadata?.usingSelection &&
    (!message.metadata.selectionPreview ||
      message.metadata.selectionPreview ===
        "Full document extraction in progress..." ||
      message.metadata.selectionPreview.includes("undefined"));

  // Get paper categories if they exist
  const paperCategories = message.metadata?.paper_categories || [];

  // More robust check for object categories
  const isObjectCategories = paperCategories.length > 0 && 
    typeof paperCategories[0] === 'object' && 
    paperCategories[0] !== null &&
    (paperCategories[0] as any).code !== undefined;
  
  // Force render of categories for debugging
  console.log("paperCategories:", paperCategories);
  console.log("isObjectCategories:", isObjectCategories);

  return (
    <div
      className={`${
        message.role === "user" ? "self-start" : "self-end"
      } rounded-lg p-3 text-white max-w-3xl`}
    >
      {message.role === "user" ? (
        <div className="flex flex-col rounded-md bg-blue-500 p-3">
          <div className="flex items-center">
            <span className="text-lg font-bold text-white">User</span>
            {message.metadata?.usingSelection && (
              <div className="ml-2 bg-white text-blue-600 text-xs px-2 py-0.5 rounded-full flex items-center">
                {isExtractionInProgress ? (
                  <>
                    <FaSpinner className="animate-spin mr-1" />
                    Extracting...
                  </>
                ) : message.metadata?.isFullDocument ? (
                  <>
                    <FaFile className="mr-1" />
                    Full Document
                  </>
                ) : (
                  <>
                    <FaPaintBrush className="mr-1" />
                    Selection
                  </>
                )}
              </div>
            )}
          </div>

          {message.metadata?.usingSelection && (
            <div
              className={`text-xs ${
                isExtractionInProgress
                  ? "bg-yellow-600"
                  : message.metadata?.isFullDocument
                  ? "bg-green-600"
                  : "bg-blue-600"
              } p-1 rounded mt-1 mb-2`}
            >
              {isExtractionInProgress ? (
                <span className="font-medium">Extracting document text...</span>
              ) : message.metadata?.isFullDocument ? (
                <span className="font-medium">
                  Using entire document ({message.metadata.selectionLength}{" "}
                  characters)
                </span>
              ) : (
                <>
                  <span className="font-medium">Selected text: </span>
                  {message.metadata.selectionPreview}
                </>
              )}
            </div>
          )}
          {message.content}
        </div>
      ) : (
        <div className="rounded-md bg-white p-3 text-black">
          {Array.isArray(paperCategories) && paperCategories.length > 0 && (
            <div className="mb-3">
              <div className="text-xs text-gray-500 mb-1">
                Paper Categories:
              </div>
              <div className="flex flex-wrap gap-1">
                {isObjectCategories ? 
                  // Handle array of category objects with code and label
                  paperCategories.map((category: any, index: number) => (
                    <span
                      key={index}
                      className="inline-flex items-center bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
                      title={category.code}
                    >
                      <FaTag className="mr-1" />
                      {category.label || category.code || "Unknown"}
                    </span>
                  ))
                : 
                  // Handle array of category strings (backward compatibility)
                  paperCategories.map((category: any, index: number) => (
                    <span
                      key={index}
                      className="inline-flex items-center bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
                    >
                      <FaTag className="mr-1" />
                      {typeof category === 'string' ? category : JSON.stringify(category)}
                    </span>
                  ))
                }
              </div>
            </div>
          )}
          <Markdown>
            {Array.isArray(message.content)
              ? message.content.join("")
              : message.content}
          </Markdown>
        </div>
      )}
    </div>
  );
}
