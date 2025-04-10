export interface MessageType {
  role: "user" | "assistant" | "system";
  content: string;
  pending?: boolean;
  metadata?: {
    usingSelection?: boolean;
    selectionLength?: number;
    selectionPreview?: string;
    isFullDocument?: boolean;
    paper_categories?: Array<string | { code: string; label: string }>;
    [key: string]: unknown;
  };
}

export const MOCK_MESSAGES: MessageType[] = [
  {
    role: "user",
    content: "Hello, how are you?",
  },
  {
    role: "assistant",
    content: "I am fine, thank you!",
  },
  {
    role: "user",
    content: "What is your name?",
  },
  {
    role: "assistant",
    content: "I am a chatbot.",
  },
];
