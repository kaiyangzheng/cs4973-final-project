import { ReactElement } from "react";
import { MessageType } from "./types";

export default function Message({
  message,
}: {
  message: MessageType;
}): ReactElement {
  return (
    <div
      className={`${
        message.role === "user" ? "self-start" : "self-end"
      } rounded-lg p-3 text-white`}
    >
      {message.role === "user" ? (
        <div className="flex flex-col rounded-md bg-blue-500 p-3">
          <span className="text-lg font-bold text-white">User</span>
          {message.content}
        </div>
      ) : (
        <div className="rounded-md bg-white p-3 text-black">
          {message.content}
        </div>
      )}
    </div>
  );
}
