import api from "./config";

const SERVER_URL = import.meta.env.SERVER_URL;
const QUERY_API_URL = `${SERVER_URL}/api/queries/`;

export interface QueryRequest {
  prompt: string;
  paper_content?: string;
  socket_id?: string;
}

// Define response interfaces
interface QueryResponse {
  success: boolean;
  response: string;
  pending: boolean;
  paper_categories?: string[];
}

// Define a Query interface based on server response
export interface Query {
  id: number;
  prompt: string;
  response: string;
  pending: boolean;
  created_at: string;
  updated_at: string;
  paper_categories?: string[];
}

const addQuery = async (request: QueryRequest): Promise<QueryResponse> => {
  const res = await api.post(QUERY_API_URL, {
    request,
  });

  if (res.status !== 200) {
    throw new Error("Error adding query");
  }

  return res.data;
};

const getQueries = async () => {
  const res = await api.get(QUERY_API_URL);

  if (res.status !== 200) {
    throw new Error("Error getting queries");
  }

  return res.data;
};

export { addQuery, getQueries };
