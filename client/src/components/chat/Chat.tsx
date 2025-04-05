import { useEffect, useState, useRef, type ReactElement } from "react";
import { FaArrowUp, FaFile, FaPaintBrush, FaRegFileAlt } from "react-icons/fa";
import Message from "./Message";
import { MessageType } from "./types";
import { usePdfContext } from "../../hooks/usePdf";
import { QueryRequest } from "../../services/queryService";
import { useSocket } from "../../hooks/useSocket";

export default function Chat(): ReactElement {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [promptInput, setPromptInput] = useState("");
  const { socket } = useSocket();
  const { file, selectedText, allText, handleClearSelections } =
    usePdfContext();

  // Add ref for the chat container
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Function to scroll to bottom of chat
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    socket.on("query_response", (response) => {
      setMessages((prevMessages) => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.pending) {
          return [
            ...prevMessages.slice(0, -1),
            {
              ...lastMessage,
              content: response.response,
              pending: false,
              metadata: {
                ...lastMessage.metadata,
                paper_categories: response.paper_categories,
              },
            },
          ];
        }
        return prevMessages;
      });

      // Scroll to bottom when receiving a response
      setTimeout(scrollToBottom, 0);
    });

    return () => {
      socket.off("query_response");
    };
  }, [socket]);

  // Add effect to scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isFullDocumentSelected = selectedText === allText;

  const handlePromptSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!promptInput || promptInput.trim() === "" || !file || !selectedText) {
      return;
    }

    const newMessage: MessageType = {
      role: "user",
      content: promptInput,
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    // Scroll to bottom immediately after adding user message
    scrollToBottom();

    // TODO: call addQuery API and await a response to populate the message
    const addQueryRequest: QueryRequest = {
      prompt: promptInput,
      paper_content: isFullDocumentSelected ? allText : selectedText,
      socket_id: socket.id,
    };

    const response = await fetch("http://localhost:8000/api/queries/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(addQueryRequest),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch data");
    }

    const data = await response.json();

    const newResponseMessage: MessageType = {
      role: "assistant",
      content: "Thinking...",
      pending: true,
      metadata: {
        usingSelection: !isFullDocumentSelected,
        selectionLength: selectedText.length,
        selectionPreview: selectedText.slice(0, 50),
        isFullDocument: isFullDocumentSelected,
        paper_categories: data.paper_categories,
      },
    };
    setMessages((prevMessages) => [...prevMessages, newResponseMessage]);

    // Scroll to bottom again after adding assistant message
    scrollToBottom();

    setPromptInput("");
  };

  return (
    <>
      <div className="flex h-screen flex-col bg-gray-600">
        <div className="flex h-16 items-center justify-center text-white px-4">
          <h1 className="text-xl font-bold">Research Paper Assistant</h1>
          {file && (
            <div className="ml-4 text-sm bg-blue-500 px-2 py-1 rounded-full flex items-center">
              <FaRegFileAlt className="mr-1" /> {file.name}
            </div>
          )}
        </div>

        {/* Add ref to the chat container */}
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4">
          {messages.map((message: MessageType, index) => (
            <Message key={index} message={message} />
          ))}
        </div>

        {selectedText && (
          <div className="px-3 py-2 bg-blue-100 border-t border-blue-300">
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                {isFullDocumentSelected ? (
                  <>
                    <FaFile className="text-blue-500 mr-1" />
                    <span className="text-sm font-medium text-blue-800">
                      Using entire document ({selectedText.length} characters)
                    </span>
                  </>
                ) : (
                  <>
                    <FaPaintBrush className="text-blue-500 mr-1" />
                    <span className="text-sm font-medium text-blue-800">
                      Using selected text ({selectedText.length} characters)
                    </span>
                  </>
                )}
              </div>
              <button
                onClick={handleClearSelections}
                className="text-xs text-blue-600 hover:text-blue-800 underline cursor-pointer"
              >
                Clear selection
              </button>
            </div>
          </div>
        )}

        <div className="p-3">
          <form
            className="relative flex items-center"
            onSubmit={handlePromptSubmit}
          >
            <input
              type="text"
              placeholder={
                isFullDocumentSelected
                  ? "Ask about the entire document..."
                  : "Ask about the selected text......"
              }
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              className="flex-1 rounded-lg border border-gray-500 p-5 pr-16 text-black focus:outline-none bg-white"
            />
            <button
              type="submit"
              className="absolute right-4 top-1/2 -translate-y-1/2 rounded-lg"
            >
              <FaArrowUp />
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
