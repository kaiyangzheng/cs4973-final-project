export interface MessageType {
  role: "user" | "assistant";
  content: string;
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
