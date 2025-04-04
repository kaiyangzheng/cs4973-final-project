import { useState, type ReactElement, useEffect, useRef } from 'react';
import { FaArrowUp, FaRegFileAlt, FaPaintBrush, FaFile } from 'react-icons/fa';
import Message from './Message';
import { MessageType } from './types';
import { addQuery, getQueries, type Query } from '../../services/queryService';
import { usePdfContext } from '../../hooks/usePdfContext';
import React from 'react';

export default function Chat(): ReactElement {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [promptInput, setPromptInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const { 
    selectedText, 
    pdfContent, 
    setPdfContent, 
    isSelectionActive, 
    clearSelection,
    pdfFileName,
    selectionInfo,
    extractAllText,
    setSelectedText,
    setSelectionInfo
  } = usePdfContext();

  // Determine if the selection is a full document selection
  const isFullDocumentSelected = isSelectionActive && selectionInfo?.pageNumber === 0;

  // Function to check if response indicates PDF processing issues
  const checkForPdfIssues = (response: string): boolean => {
    // Only check for actual error messages, not descriptive content
    const pdfErrorKeywords = [
      "⚠️ The uploaded PDF couldn't be processed",
      "Failed to extract text from PDF",
      "Error: Could not process PDF file",
      "PDF extraction failed",
      "Invalid PDF format"
    ];
    
    // Only consider it an error if:
    // 1. It contains an error keyword
    // 2. It's a short message (error messages are typically short)
    // 3. It's not a valid paper summary (which would be longer and contain paper content)
    return pdfErrorKeywords.some(keyword => response.includes(keyword)) &&
           response.length < 500 &&
           !response.includes("**Title:**") &&
           !response.includes("**Authors:**") &&
           !response.includes("**Summary:**");
  };

  useEffect(() => {
    const pollInterval = 10000; // 10 seconds
    const MAX_POLLS = 60; // 10 minutes of polling
    let pollCount = 0;
    
    // Reset poll count when messages change and there are pending messages
    if (messages.some(msg => msg.role === 'assistant' && (msg.content === 'Thinking...' || msg.pending === true))) {
      pollCount = 0;
    }
    
    const checkPendingMessages = async () => {
      try {
        const fetchedQueries = await getQueries();
        console.log("Received queries:", fetchedQueries);
        
        // Sort queries by ID in descending order to prioritize newest queries
        const sortedQueries = [...fetchedQueries].sort((a, b) => b.id - a.id);
        setMessages(prevMessages => {
          const newMessages = [...prevMessages];
          
          // Check for any pending messages
          for (let i = newMessages.length - 1; i >= 0; i--) {
            const message = newMessages[i];
            
            // Skip non-assistant messages or non-pending messages
            if (message.role !== "assistant") continue;
            
            // Find the previous user message to get the prompt
            if (i > 0 && newMessages[i-1].role === "user") {
              const userMessage = newMessages[i-1];
              const userPrompt = userMessage.content.trim();
              console.log("Looking for update to prompt:", userPrompt);
              
              // Find the most recent query that matches the user prompt
              const matchingQueries = sortedQueries.filter((q: Query) => {
                const queryPrompt = q.prompt.trim();
                const userPromptTrimmed = userPrompt.trim();
                console.log(`Comparing: "${queryPrompt}" (${q.id}) vs "${userPromptTrimmed}" - exact match: ${queryPrompt === userPromptTrimmed}, case insensitive: ${queryPrompt.toLowerCase() === userPromptTrimmed.toLowerCase()}, pending: ${q.pending}`);
                return (
                  (queryPrompt === userPromptTrimmed || 
                   queryPrompt.toLowerCase() === userPromptTrimmed.toLowerCase()) && 
                  q.pending === false
                );
              });
              
              console.log("All matching queries:", matchingQueries.map(q => ({ id: q.id, prompt: q.prompt, created_at: q.created_at })));
              
              if (matchingQueries.length > 0) {
                // Log all matching queries with their IDs and creation times
                console.log("Matching queries before selection:", matchingQueries.map(q => 
                  `ID: ${q.id}, Created: ${q.created_at}, Prompt: "${q.prompt.substring(0, 30)}..."`
                ));
                
                const mostRecentQuery = matchingQueries.reduce((latest, current) => {
                  console.log(`Comparing queries - Current ID: ${current.id}, Latest ID: ${latest.id}, Using: ${current.id > latest.id ? 'current' : 'latest'}`);
                  return current.id > latest.id ? current : latest;
                });
                
                console.log("Selected most recent query ID:", mostRecentQuery.id, "created at:", mostRecentQuery.created_at);
                console.log("Found updated response for:", userPrompt);
                console.log("Response:", mostRecentQuery.response);
                console.log("Paper categories from query:", mostRecentQuery.paper_categories);
                
                // Update the message with the response
                newMessages[i] = {
                  role: "assistant",
                  content: mostRecentQuery.response,
                  pending: false,
                  metadata: {
                    ...(message.metadata || {}),
                    paper_categories: mostRecentQuery.paper_categories || []
                  }
                };
                console.log("Updated message metadata:", newMessages[i].metadata);
              } else {
                // Check if there's a pending query for this prompt
                const pendingQuery = sortedQueries.find((q: Query) => {
                  const queryPrompt = q.prompt.trim();
                  return (
                    (queryPrompt === userPrompt || 
                     queryPrompt.toLowerCase() === userPrompt.toLowerCase()) && 
                    q.pending === true
                  );
                });
                
                // Instead of setting "No response", keep showing "Thinking..." 
                if (!pendingQuery && message.content === "Thinking...") {
                  console.log("No matching query found, but keeping 'Thinking...' state for:", userPrompt);
                  // Keep the "Thinking..." message
                  newMessages[i] = {
                    role: "assistant",
                    content: "Thinking...",
                    pending: true
                  };
                }
              }
            }
          }
          
          return newMessages;
        });
      } catch (error) {
        console.error("Error fetching queries:", error);
      }
    };
    
    const interval = setInterval(async () => {
      // Only poll if there are pending messages and we haven't reached the max
      if (messages.some(msg => msg.role === 'assistant' && (msg.content === 'Thinking...' || msg.pending === true)) && pollCount < MAX_POLLS) {
        console.log("Checking for updates to pending messages...");
        await checkPendingMessages();
        pollCount++;
        
        // If we've reached the max polls, show a message
        if (pollCount === MAX_POLLS) {
          console.log("Reached maximum number of polling attempts");
          setMessages(prevMessages => {
            const newMessages = [...prevMessages];
            // Update any remaining "Thinking..." messages
            for (let i = 0; i < newMessages.length; i++) {
              if (newMessages[i].role === "assistant" && (newMessages[i].content === "Thinking..." || newMessages[i].pending === true)) {
                newMessages[i] = {
                  role: "assistant",
                  content: "Your request is still processing. You can continue asking questions or check back later for results.",
                  pending: false
                };
              }
            }
            return newMessages;
          });
        }
      }
    }, pollInterval);
    
    // Run an immediate check if we have pending messages
    if (messages.some(msg => msg.role === 'assistant' && (msg.content === 'Thinking...' || msg.pending === true))) {
      checkPendingMessages();
    }
    
    return () => clearInterval(interval);
  }, [messages]);

  const handlePromptSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptInput.trim()) return;

    // Check if we have selected text or need to extract from PDF
    const hasSelection = Boolean(selectedText && selectedText.trim().length > 0);
    const hasPdfContent = Boolean(pdfContent); // PDF content could be ArrayBuffer or other non-string type
    
    // Create user message first
    const userMessage: MessageType = {
      role: "user",
      content: promptInput,
        metadata: {
            usingSelection: hasSelection,
            selectionLength: selectedText?.length || 0,
            selectionPreview: hasSelection ? selectedText?.substring(0, 100) + "..." : undefined
        }
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Add temporary assistant message
    const tempAssistantMessage: MessageType = {
      role: "assistant",
      content: "Thinking...",
        pending: true
    };
    setMessages(prev => [...prev, tempAssistantMessage]);
    
    // If no selection but we have PDF content, try to extract text
    let contentToSend: string = selectedText || "";
    if (!hasSelection && hasPdfContent) {
        try {
            // Try to extract text from PDF
            const extractedText = await extractAllText();
            if (extractedText && typeof extractedText === 'string') {
                // Check if the text looks like PDF binary data (starts with %PDF)
                if (extractedText.startsWith('%PDF')) {
                    console.log("Detected PDF binary data instead of text. Cannot extract content properly.");
                    setMessages(prevMessages => [
                        ...prevMessages,
                        {
                            role: "system",
                            content: "Unable to extract text from this PDF. The file may be scanned, encrypted, or in a format that cannot be processed. Try selecting specific text or uploading a different PDF."
                        }
                    ]);
                    return; // Don't proceed with sending the request
                }
                
                contentToSend = extractedText;
                setSelectedText(extractedText);
                setSelectionInfo({
                    pageNumber: 1,
                    length: extractedText.length
                });
            } else {
                console.log("Failed to extract text from PDF");
                setMessages(prevMessages => [
                    ...prevMessages,
                    {
                        role: "system",
                        content: "Unable to extract text from this PDF. Try selecting specific text or uploading a different PDF."
                    }
                ]);
                return; // Don't proceed with sending the request
            }
        } catch (error) {
            console.error("Error extracting text:", error);
            setMessages(prevMessages => [
                ...prevMessages,
                {
                    role: "system",
                    content: "Error extracting text from PDF. Try selecting specific text instead."
                }
            ]);
            return; // Don't proceed with sending the request
        }
    }

    // If still no content, show an error instead of trying to select all text
    if (!contentToSend) {
        setMessages(prevMessages => [
            ...prevMessages,
            {
                role: "system",
                content: "No content to analyze. Please upload a PDF or select some text."
            }
        ]);
        return; // Don't proceed with sending the request
    }

    try {
      const requestData = {
        prompt: promptInput,
            paper_content: contentToSend || ""
        };
        
        console.log("Sending request with data:", requestData);

        const response = await fetch("http://localhost:8000/api/queries/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Received response data:", data);
        console.log("Paper categories from response:", data.paper_categories);
        console.log("Response ID:", data.id, "Created at:", data.created_at);
        console.log("Full response text:", data.response?.substring(0, 100) + "...");

        // Update the temporary assistant message with the actual response
        setMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === "assistant" && lastMessage.content === "Thinking...") {
                console.log("Creating new message with categories:", data.paper_categories);
                const newMessage: MessageType = {
          role: "assistant",
                    content: data.response || lastMessage.content,
                    pending: data.pending || false,
                    metadata: {
                        ...(lastMessage.metadata || {}),
                        paper_categories: data.paper_categories || []
                    }
                };
                console.log("New message metadata:", newMessage.metadata);
                newMessages[newMessages.length - 1] = newMessage;
            }
        return newMessages;
      });

        // Clear the input
        setPromptInput("");
        
        // Scroll to bottom
        chatContainerRef.current?.scrollTo({
            top: chatContainerRef.current.scrollHeight,
            behavior: "smooth",
      });
    } catch (error) {
        console.error("Error sending query:", error);
        // Update the temporary assistant message with the error
        setMessages(prev => {
            const newMessages = [...prev];
            const lastMessage = newMessages[newMessages.length - 1];
            if (lastMessage.role === "assistant" && lastMessage.content === "Thinking...") {
                const newMessage: MessageType = {
          role: "assistant",
                    content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
                    pending: false
        };
                newMessages[newMessages.length - 1] = newMessage;
            }
        return newMessages;
      });
    }
  };

  return (
    <React.Fragment>
      <div className="flex h-screen flex-col bg-gray-600">
        <div className="flex h-16 items-center justify-center text-white">
          <h1 className="text-xl font-bold">Research Paper Assistant</h1>
          {pdfFileName && (
            <div className="ml-4 text-sm bg-blue-500 px-2 py-1 rounded-full flex items-center">
              <FaRegFileAlt className="mr-1" /> {pdfFileName}
            </div>
          )}
        </div>
        
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4">
          {messages.map((message: MessageType, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        
        {isSelectionActive && selectedText && (
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
                onClick={clearSelection}
                className="text-xs text-blue-600 hover:text-blue-800 underline"
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
              placeholder={isFullDocumentSelected
                ? "Ask about the entire document..."
                : isSelectionActive 
                  ? "Ask about the selected text..." 
                  : pdfContent 
                    ? "Ask questions about this PDF..." 
                    : "Upload a PDF first or ask a general question..."
              }
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              className="flex-1 rounded-lg border border-gray-500 p-5 pr-16 text-black focus:outline-none bg-white"
              disabled={isLoading}
            />
            <button
              type="submit"
              className={`absolute right-4 top-1/2 -translate-y-1/2 rounded-lg ${
                isLoading 
                  ? "bg-gray-500" 
                  : isFullDocumentSelected || (pdfContent && !isSelectionActive)
                    ? "bg-green-500 hover:bg-green-600" 
                    : isSelectionActive 
                      ? "bg-blue-500 hover:bg-blue-600" 
                      : "bg-blue-500 hover:bg-blue-600"
              } px-2 py-2 text-white`}
              disabled={isLoading}
            >
              <FaArrowUp />
            </button>
          </form>
        </div>
      </div>
    </React.Fragment>
  );
}
