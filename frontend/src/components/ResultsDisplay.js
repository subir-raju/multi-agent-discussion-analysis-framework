import React from "react";
import {
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
  Paper,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

const ResultsDisplay = ({ results }) => {
  if (!results) return null;

  const {
    influence_metrics,
    sentiment_metrics,
    participation_metrics,
    coherence_scores,
    topic_shifts,
    visualizations,
    summary_text,
  } = results;

  const renderMetricValue = (value) => {
    if (typeof value === "number") {
      return value.toFixed(3);
    }
    return value;
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Analysis Results
      </Typography>

      {/* Summary Section */}
      <Card
        sx={{ mb: 3, backgroundColor: "#e8f5e8", border: "2px solid #4CAF50" }}
      >
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Summary
          </Typography>
          <Typography variant="body1">{summary_text}</Typography>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Visualizations
          </Typography>
          <Grid container spacing={3}>
            {Object.entries(visualizations).map(([key, imageData]) => (
              <Grid item xs={12} md={6} key={key}>
                <Paper elevation={2} sx={{ p: 2 }}>
                  <Typography
                    variant="subtitle1"
                    gutterBottom
                    sx={{ textTransform: "capitalize" }}
                  >
                    {key.replace(/_/g, " ")}
                  </Typography>
                  <img
                    src={imageData}
                    alt={key}
                    style={{
                      width: "100%",
                      height: "auto",
                      border: "1px solid #ddd",
                      borderRadius: "4px",
                    }}
                  />
                </Paper>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6"> Influence Metrics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Agent Influence Count
              </Typography>
              {Object.entries(influence_metrics.agent_influence_count).map(
                ([agent, count]) => (
                  <Box
                    key={agent}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{agent}:</Typography>
                    <Chip label={count} size="small" color="primary" />
                  </Box>
                )
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Influence Pairs
              </Typography>
              {Object.entries(influence_metrics.influence_pairs).map(
                ([pair, count]) => (
                  <Box
                    key={pair}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{pair}:</Typography>
                    <Chip label={count} size="small" color="secondary" />
                  </Box>
                )
              )}
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6"> Sentiment & Topic Metrics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Average Sentiment per Agent
              </Typography>
              {Object.entries(sentiment_metrics.avg_sentiment_per_agent).map(
                ([agent, score]) => (
                  <Box
                    key={agent}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{agent}:</Typography>
                    <Chip
                      label={renderMetricValue(score)}
                      size="small"
                      color={
                        score > 0.05
                          ? "success"
                          : score < -0.05
                          ? "error"
                          : "default"
                      }
                    />
                  </Box>
                )
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Average Topic Relevance per Agent
              </Typography>
              {Object.entries(
                sentiment_metrics.avg_topic_relevance_per_agent
              ).map(([agent, score]) => (
                <Box
                  key={agent}
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mb: 1,
                  }}
                >
                  <Typography>{agent}:</Typography>
                  <Chip
                    label={renderMetricValue(score)}
                    size="small"
                    color="info"
                  />
                </Box>
              ))}
            </Grid>
          </Grid>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Turn-by-Turn Sentiment Labels
          </Typography>
          {results.dialogues &&
            results.sentiment_metrics &&
            results.sentiment_metrics.sentiment_labels &&
            results.sentiment_metrics.sentiment_labels.length ===
              results.dialogues.length &&
            results.sentiment_metrics.sentiment_labels.map((label, index) => (
              <Box key={index} sx={{ mb: 0.5 }}>
                Turn {index + 1} ({results.dialogues[index].agent}): {label}
              </Box>
            ))}

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
            {sentiment_metrics.sentiment_labels.map((label, index) => (
              <Chip
                key={index}
                label={`Turn ${index + 1}: ${label}`}
                size="small"
                color={
                  label === "positive"
                    ? "success"
                    : label === "negative"
                    ? "error"
                    : "default"
                }
              />
            ))}
          </Box>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Participation Metrics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Turn Counts
              </Typography>
              {Object.entries(participation_metrics.turn_counts).map(
                ([agent, count]) => (
                  <Box
                    key={agent}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{agent}:</Typography>
                    <Chip label={count} size="small" color="primary" />
                  </Box>
                )
              )}
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Word Counts
              </Typography>
              {Object.entries(participation_metrics.word_counts).map(
                ([agent, count]) => (
                  <Box
                    key={agent}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{agent}:</Typography>
                    <Chip label={count} size="small" color="secondary" />
                  </Box>
                )
              )}
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Avg Words per Turn
              </Typography>
              {Object.entries(participation_metrics.avg_words_per_turn).map(
                ([agent, avg]) => (
                  <Box
                    key={agent}
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      mb: 1,
                    }}
                  >
                    <Typography>{agent}:</Typography>
                    <Chip
                      label={renderMetricValue(avg)}
                      size="small"
                      color="info"
                    />
                  </Box>
                )
              )}
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6"> Coherence Analysis</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="subtitle2" gutterBottom>
            Coherence Scores Between Consecutive Turns
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
            {coherence_scores.map((score, index) => (
              <Chip
                key={index}
                label={`${index + 1}→${index + 2}: ${renderMetricValue(score)}`}
                size="small"
                color={
                  score >= 0.7 ? "success" : score >= 0.3 ? "warning" : "error"
                }
              />
            ))}
          </Box>
        </AccordionDetails>
      </Accordion>

      {topic_shifts.length > 0 && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Topic Shifts Detected</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {topic_shifts.map((shift, index) => (
              <Paper key={index} elevation={2} sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {shift.shift_display}
                </Typography>
                <Typography variant="body2">
                  <b>From:</b> {shift.from_keywords.join(", ")}
                  {typeof shift.from_topic_relevance !== "undefined" && (
                    <>
                      {" "}
                      (Relevance:{" "}
                      {renderMetricValue(shift.from_topic_relevance)})
                    </>
                  )}
                  {typeof shift.from_discussion_relevance !== "undefined" && (
                    <>
                      <br />
                      <b>From (Relevance to Discussion Topic):</b>{" "}
                      {renderMetricValue(shift.from_discussion_relevance)}
                    </>
                  )}
                  <br />
                  <b>To:</b> {shift.to_keywords.join(", ")}
                  {typeof shift.to_topic_relevance !== "undefined" && (
                    <>
                      {" "}
                      (Relevance: {renderMetricValue(shift.to_topic_relevance)})
                    </>
                  )}
                  {typeof shift.to_discussion_relevance !== "undefined" && (
                    <>
                      <br />
                      <b>To (Relevance to Discussion Topic):</b>{" "}
                      {renderMetricValue(shift.to_discussion_relevance)}
                    </>
                  )}
                  <br />
                  <b>Semantic Similarity:</b>{" "}
                  {renderMetricValue(shift.semantic_similarity)}
                  <br />
                  <b>Topic Shift Magnitude:</b>{" "}
                  {renderMetricValue(shift.shift_magnitude)}
                </Typography>
              </Paper>
            ))}
          </AccordionDetails>
        </Accordion>
      )}

      {/* Topic Credibility Analysis */}
      {results.topic_credibility_metrics && (
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ fontWeight: "bold" }}>
              Topic Credibility Analysis
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {/* Agent Credibility Scores */}
              {results.topic_credibility_metrics.agent_credibility_scores && (
                <Grid item xs={12} md={6}>
                  <Typography
                    variant="subtitle1"
                    fontWeight="bold"
                    gutterBottom
                  >
                    Agent Credibility Scores
                  </Typography>
                  {Object.entries(
                    results.topic_credibility_metrics.agent_credibility_scores
                  ).map(([agent, score]) => (
                    <Box
                      key={agent}
                      sx={{ display: "flex", alignItems: "center", mb: 1 }}
                    >
                      <Typography variant="body2" sx={{ minWidth: 80 }}>
                        {agent}:
                      </Typography>
                      <Chip
                        label={`${renderMetricValue(score)} ${
                          score >= 0.7
                            ? "(Excellent)"
                            : score >= 0.5
                            ? "(Good)"
                            : score >= 0.3
                            ? "(Average)"
                            : "(Poor)"
                        }`}
                        color={
                          score >= 0.7
                            ? "success"
                            : score >= 0.5
                            ? "primary"
                            : score >= 0.3
                            ? "warning"
                            : "error"
                        }
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    </Box>
                  ))}
                </Grid>
              )}

              {/* Credibility Ranking */}
              {results.topic_credibility_metrics.credibility_ranking && (
                <Grid item xs={12} md={6}>
                  <Typography
                    variant="subtitle1"
                    fontWeight="bold"
                    gutterBottom
                  >
                    Credibility Ranking
                  </Typography>
                  {results.topic_credibility_metrics.credibility_ranking.map(
                    (entry, index) => (
                      <Box
                        key={index}
                        sx={{ display: "flex", alignItems: "center", mb: 1 }}
                      >
                        <Typography variant="body2" sx={{ minWidth: 40 }}>
                          #{entry.rank}
                        </Typography>
                        <Typography variant="body2" sx={{ minWidth: 80 }}>
                          {entry.agent}
                        </Typography>
                        <Chip
                          label={`${
                            entry.credibility_level
                          } (${renderMetricValue(entry.credibility_score)})`}
                          color={
                            entry.credibility_level === "Excellent"
                              ? "success"
                              : entry.credibility_level === "Good"
                              ? "primary"
                              : entry.credibility_level === "Average"
                              ? "warning"
                              : "error"
                          }
                          size="small"
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    )
                  )}
                </Grid>
              )}

              {/* Detailed Breakdown */}
              {results.topic_credibility_metrics.credibility_ranking && (
                <Grid item xs={12}>
                  <Typography
                    variant="subtitle1"
                    fontWeight="bold"
                    gutterBottom
                  >
                    Detailed Component Breakdown
                  </Typography>
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                    {results.topic_credibility_metrics.credibility_ranking.map(
                      (entry, index) => (
                        <Paper key={index} sx={{ p: 2, minWidth: 200 }}>
                          <Typography
                            variant="subtitle2"
                            fontWeight="bold"
                            color="primary"
                          >
                            {entry.agent}
                          </Typography>
                          <Typography variant="body2">
                            Topic Relevance:{" "}
                            {renderMetricValue(entry.topic_relevance)}
                          </Typography>
                          <Typography variant="body2">
                            Consistency: {renderMetricValue(entry.consistency)}
                          </Typography>
                          <Typography variant="body2">
                            Semantic Depth: {renderMetricValue(entry.depth)}
                          </Typography>
                          <Typography variant="body2">
                            Off-topic Penalty:{" "}
                            {renderMetricValue(entry.off_topic_penalty)}
                          </Typography>
                        </Paper>
                      )
                    )}
                  </Box>
                </Grid>
              )}
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
};

export default ResultsDisplay;
