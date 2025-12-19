import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const analyzeDialogue = async (dialogueData) => {
  try {
    const response = await api.post("/analyze", dialogueData);
    return response.data;
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get("/health");
    return response.data;
  } catch (error) {
    console.error("Health check failed:", error);
    throw error;
  }
};
