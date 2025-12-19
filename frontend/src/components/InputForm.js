import React, { useState } from "react";
import { TextField, Button, Typography, Box, Grid, Paper } from "@mui/material";
import { Card } from "@mui/material";

const InputForm = ({ onSubmit, loading }) => {
  const [numAgents, setNumAgents] = useState("");
  const [agentNames, setAgentNames] = useState([]);
  const [turnsPerAgent, setTurnsPerAgent] = useState("");
  const [topic, setTopic] = useState("");
  const [dialogues, setDialogues] = useState([]);
  const [showDialogueInputs, setShowDialogueInputs] = useState(false);

  // Handle change in number of agents AND adjust agentName boxes accordingly
  const handleNumAgentsChange = (e) => {
    const value = e.target.value;
    setNumAgents(value);

    const n = parseInt(value, 10) || 0;
    setAgentNames((prev) => {
      // Truncate or expand agentNames array
      const arr = prev.slice(0, n);
      while (arr.length < n) arr.push("");
      return arr;
    });
  };

  // Handle editing of agent names
  const handleAgentNameChange = (idx, val) => {
    setAgentNames((prev) => {
      const copy = [...prev];
      copy[idx] = val;
      return copy;
    });
  };

  // Handle number of turns per agent input
  const handleTurnsChange = (e) => {
    setTurnsPerAgent(e.target.value);
  };

  // When the user clicks "Set Up Dialogue Structure"
  const handleAgentsAndTurnsSubmit = () => {
    const agents = parseInt(numAgents, 10);
    const turns = parseInt(turnsPerAgent, 10);

    if (agents > 0 && turns > 0 && agentNames.every((n) => n.trim())) {
      // Initialize empty dialogues structure
      const newDialogues = [];
      for (let i = 0; i < agents * turns; i++) {
        newDialogues.push({ agent: "", text: "" });
      }
      setDialogues(newDialogues);
      setShowDialogueInputs(true);
    }
  };

  const handleDialogueAgentChange = (idx, value) => {
    setDialogues((prev) => {
      const arr = [...prev];
      arr[idx].agent = value;
      return arr;
    });
  };

  // Handle actual dialogue text input by user
  const handleDialogueChange = (idx, value) => {
    setDialogues((prev) => {
      const arr = [...prev];
      arr[idx].text = value;
      return arr;
    });
  };

  // Only "Analyze" if all texts present
  const isFormValid = () =>
    topic.trim() &&
    agentNames.length > 0 &&
    agentNames.every((n) => n.trim()) &&
    dialogues.length > 0 &&
    dialogues.every((d) => d.text.trim());

  // When user clicks "Analyze Dialogue"
  const handleAnalyze = () => {
    const analysisData = {
      topic,
      num_agents: parseInt(numAgents, 10),
      agent_names: agentNames.map((n) => n.trim()),
      turns_per_agent: parseInt(turnsPerAgent, 10),
      dialogues: dialogues.map((d) => ({
        agent: d.agent,
        text: d.text,
      })),
    };
    onSubmit(analysisData);
  };

  return (
    <Box maxWidth={900} mx="auto">
      <Typography variant="h4" mt={2} mb={2} align="center">
        Multi-Agent Discussion Analysis Framework
      </Typography>
      <Typography variant="subtitle1" mb={3} align="center">
        Turn AI-generated multi-agent discussions into interpretable insights on
        sentiment, topics, coherence, and influence
      </Typography>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        {/* Step 1: Set Agent Count and Names */}
        <Typography variant="h6" gutterBottom>
          Step 1: Configuration
        </Typography>
        <TextField
          type="number"
          label="Number of Agents"
          value={numAgents}
          onChange={handleNumAgentsChange}
          inputProps={{ min: 2, max: 10 }}
          sx={{ mb: 2, mr: 2, width: 200 }}
        />
        {/* Dynamically show agent name inputs */}
        {parseInt(numAgents, 10) > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1">Enter agent names:</Typography>
            <Grid container spacing={1}>
              {Array.from({ length: parseInt(numAgents, 10) }).map((_, idx) => (
                <Grid item key={idx}>
                  <TextField
                    label={`Agent ${idx + 1} Name`}
                    value={agentNames[idx] || ""}
                    onChange={(e) => handleAgentNameChange(idx, e.target.value)}
                    variant="outlined"
                    size="small"
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
        <TextField
          type="number"
          label="Number of Turns per Agent"
          value={turnsPerAgent}
          onChange={handleTurnsChange}
          inputProps={{ min: 1, max: 20 }}
          sx={{ mb: 2, width: 200 }}
        />
        <Box>
          <Button
            variant="contained"
            sx={{ mt: 1 }}
            disabled={
              !numAgents ||
              parseInt(numAgents, 10) < 2 ||
              !turnsPerAgent ||
              parseInt(turnsPerAgent, 10) < 1 ||
              agentNames.length !== parseInt(numAgents, 10) ||
              agentNames.some((n) => !n.trim())
            }
            onClick={handleAgentsAndTurnsSubmit}
          >
            Set Up Dialogue Structure
          </Button>
        </Box>
      </Paper>

      {/* Step 2: Topic specification */}
      {showDialogueInputs && (
        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Step 2: Discussion Topic
          </Typography>
          <TextField
            label="Discussion Topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            variant="outlined"
            multiline
            rows={2}
            fullWidth
            placeholder="e.g., The impact of smartphones on human interaction"
          />
        </Paper>
      )}

      {/* Step 3: Enter Dialogue */}
      {showDialogueInputs && (
        <Box sx={{ mt: 3, width: "100%" }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Step 3: Enter Dialogue Turns (in order)
          </Typography>
          <Grid container direction="column" spacing={2}>
            {dialogues.map((d, idx) => (
              <Grid
                container
                direction="column"
                spacing={1}
                key={idx}
                sx={{ mb: 2 }}
              >
                <Grid item>
                  <select
                    value={d.agent}
                    onChange={(e) =>
                      handleDialogueAgentChange(idx, e.target.value)
                    }
                    style={{
                      width: "200px",
                      minHeight: "40px",
                      fontSize: "1rem",
                      borderRadius: "5px",
                      paddingLeft: "7px",
                      boxSizing: "border-box",
                    }}
                  >
                    <option value="" disabled>
                      Select Agent
                    </option>
                    {agentNames.map((name, i) => (
                      <option key={i} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </Grid>
                <Grid item>
                  <TextField
                    value={d.text}
                    onChange={(e) => handleDialogueChange(idx, e.target.value)}
                    label={`Turn ${idx + 1} dialogue`}
                    variant="outlined"
                    multiline
                    minRows={2}
                    fullWidth
                    sx={{ width: "100%" }}
                    placeholder={`What does ${d.agent || "the agent"} say?`}
                  />
                </Grid>
              </Grid>
            ))}

            {/* Analyze Button */}
            <Grid item sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                disabled={!isFormValid() || loading}
                onClick={handleAnalyze}
              >
                {loading ? "Analyzing..." : "Analyze Dialogue"}
              </Button>
            </Grid>
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default InputForm;
