import React, { useState } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import {
  CssBaseline,
  Container,
  Alert,
  CircularProgress,
  Box,
} from "@mui/material";
import InputForm from "./components/InputForm";
import ResultsDisplay from "./components/ResultsDisplay";
//import { analyzeDialogue } from "./services/api";

const theme = createTheme({
  palette: {
    primary: {
      main: "#4CAF50",
    },
    secondary: {
      main: "#2196F3",
    },
  },
});

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (dialogueData) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      console.log("Submitting dialogue data:", dialogueData);
      const analysisResults = await analyzeDialogue(dialogueData);
      console.log("Analysis results:", analysisResults);
      setResults(analysisResults);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(
        err.response?.data?.detail ||
          err.message ||
          "An error occurred while analyzing the dialogue"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="xl">
        <InputForm onSubmit={handleSubmit} loading={loading} />

        {loading && (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            my={4}
          >
            <CircularProgress size={60} />
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {results && <ResultsDisplay results={results} />}
      </Container>
    </ThemeProvider>
  );
}

export default App;
