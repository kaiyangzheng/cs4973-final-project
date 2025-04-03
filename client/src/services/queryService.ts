import api from "./config";

// Use direct URL with trailing slash to avoid redirects
// Include all possible client ports
const QUERY_API_URL = `http://localhost:8000/api/queries/`;

interface QueryRequest {
  prompt: string;
  paper_content?: string;
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
  console.log("Sending query to:", QUERY_API_URL);
  const res = await api.post(QUERY_API_URL, request);

  if (res.status !== 200) {
    throw new Error("Error adding query");
  }

  console.log("Query response:", res.data);
  return res.data;
};

const getQueries = async (): Promise<Query[]> => {
  console.log("Getting queries from:", QUERY_API_URL);
  try {
    const res = await api.get(QUERY_API_URL);

    if (res.status !== 200) {
      console.error("Error getting queries, status:", res.status);
      throw new Error("Error getting queries");
    }

    console.log("Queries response:", res.data);
    
    // Check that data is an array
    if (!Array.isArray(res.data)) {
      console.error("Expected array in response, got:", typeof res.data);
      return [];
    }
    
    return res.data;
  } catch (error) {
    console.error("Error fetching queries:", error);
    return [];
  }
};

export { addQuery, getQueries };
