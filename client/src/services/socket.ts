import { io } from "socket.io-client";

const socket = io("http://localhost:8000/", {
  autoConnect: false,
  transports: ["websocket"],
}).connect();

socket.on("connect", () => {
  console.log("Connected to socket server");
});

socket.on("disconnect", () => {
  console.log("Disconnected from socket server");
});

export default socket;
