import { useState, type ReactElement } from "react";
import { FaArrowUp } from "react-icons/fa";
import Message from "./Message";
import { MessageType } from "./types";

export default function Chat(): ReactElement {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [promptInput, setPromptInput] = useState("");

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!promptInput || promptInput.trim() === "") {
      return;
    }

    setTimeout(() => {
      const chatContainer = document.querySelector(".flex-1.overflow-y-auto");
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 0);

    const newMessage: MessageType = {
      role: "user",
      content: promptInput,
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    // TODO: call addQuery API and await a response to populate the message
    setPromptInput("");
  };

  return (
    <>
      <div className="flex h-screen flex-col bg-gray-600">
        <div className="flex h-16 items-center justify-center text-white">
          <h1 className="text-xl font-bold">How can I help?</h1>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {messages.map((message: MessageType, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        <div className="p-3">
          <form
            className="relative flex items-center"
            onSubmit={handlePromptSubmit}
          >
            <input
              type="text"
              placeholder="Ask a question..."
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              className="flex-1 rounded-lg border border-gray-500 p-5 pr-16 text-black focus:outline-none bg-white"
            />
            <button
              type="submit"
              className="absolute right-4 top-1/2 -translate-y-1/2 rounded-lg bg-blue-500 px-2 py-2 text-white hover:bg-blue-600"
            >
              <FaArrowUp />
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
