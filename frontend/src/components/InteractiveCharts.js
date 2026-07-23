import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell, PieChart, Pie
} from 'recharts';
import { Box, Typography, Paper, Grid } from '@mui/material';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);

const InteractiveCharts = ({ interactiveData, embeddings3d }) => {
  if (!interactiveData) return null;

  const { sentiment_trend, agent_stats } = interactiveData;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Paper sx={{ p: 1.5, border: '1px solid #ccc' }}>
          <Typography variant="body2" fontWeight="bold">{`Turn ${label}`}</Typography>
          <Typography variant="body2" color="primary">{`Agent: ${payload[0].payload.agent}`}</Typography>
          <Typography variant="body2">{`Sentiment: ${payload[0].payload.sentiment}`}</Typography>
          <Typography variant="body2" sx={{ maxWidth: 200, mt: 1, fontStyle: 'italic' }}>
            {`"${payload[0].payload.text}"`}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom align="center" sx={{ fontWeight: 'bold', mb: 4 }}>
        Interactive Analysis
      </Typography>

      <Grid container spacing={4}>
        {/* 1. Interactive Sentiment Trend */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Conversation Sentiment & Topic Relevance Trend</Typography>
            <Box sx={{ height: 400, width: '100%' }}>
              <ResponsiveContainer>
                <LineChart data={sentiment_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="turn" label={{ value: 'Turn Number', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line type="monotone" dataKey="score" stroke="#8884d8" name="Topic Relevance" strokeWidth={2} activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="coherence" stroke="#82ca9d" name="Semantic Coherence" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* 2. Agent Performance Comparison */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Agent Topic Relevance vs Sentiment</Typography>
            <Box sx={{ height: 300, width: '100%' }}>
              <ResponsiveContainer>
                <BarChart data={agent_stats}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="agent" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="relevance" fill="#0088FE" name="Avg Relevance" />
                  <Bar dataKey="avg_sentiment" fill="#00C49F" name="Avg Sentiment" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* 3. Participation Distribution */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Word Count Distribution</Typography>
            <Box sx={{ height: 300, width: '100%' }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={agent_stats}
                    dataKey="words"
                    nameKey="agent"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={(entry) => entry.agent}
                  >
                    {agent_stats.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={['#0088FE', '#00C49F', '#FFBB28', '#FF8042'][index % 4]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* 4. 3D Embedding Projector */}
        {embeddings3d && (
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>3D Conversation Embedding Space</Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Each point represents a single turn. Proximity indicates semantic similarity.
              </Typography>
              <Plot
                data={[
                  {
                    x: embeddings3d.map(p => p.x),
                    y: embeddings3d.map(p => p.y),
                    z: embeddings3d.map(p => p.z),
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {
                      size: 6,
                      color: embeddings3d.map(p => p.turn),
                      colorscale: 'Viridis',
                      opacity: 0.8
                    },
                    text: embeddings3d.map(p => `Turn ${p.turn} (${p.agent}): ${p.text}`),
                    hoverinfo: 'text'
                  }
                ]}
                layout={{
                  width: undefined,
                  height: 600,
                  autosize: true,
                  margin: { l: 0, r: 0, b: 0, t: 0 },
                  scene: {
                    xaxis: { title: 'UMAP 1' },
                    yaxis: { title: 'UMAP 2' },
                    zaxis: { title: 'UMAP 3' }
                  }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default InteractiveCharts;
