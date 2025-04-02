import api from "./config";

const SERVER_URL = import.meta.env.SERVER_URL;
const QUERY_API_URL = `${SERVER_URL}/api/queries`;

const addQuery = async (prompt: string) => {
  const res = await api.post(QUERY_API_URL, {
    prompt,
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
