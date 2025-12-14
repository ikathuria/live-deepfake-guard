import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart
} from 'recharts';
import { AnalysisResult } from '../types';

interface DetectionChartProps {
  data: AnalysisResult[];
}

export const DetectionChart: React.FC<DetectionChartProps> = ({ data }) => {
  
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload as AnalysisResult;
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-xl text-xs">
          <p className="text-slate-300 mb-1">Time: {item.timestamp.toFixed(1)}s</p>
          <p className="text-white font-bold mb-1">
            Fake Probability: <span className={item.fakeProbability > 50 ? "text-red-400" : "text-emerald-400"}>{item.fakeProbability}%</span>
          </p>
          <p className="text-slate-400 italic max-w-[200px]">{item.reasoning}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full min-h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 0,
            bottom: 0,
          }}
        >
          <defs>
            <linearGradient id="colorFake" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis 
            dataKey="timestamp" 
            stroke="#94a3b8" 
            fontSize={12} 
            tickFormatter={(val) => `${val}s`}
          />
          <YAxis 
            stroke="#94a3b8" 
            fontSize={12} 
            domain={[0, 100]} 
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" label={{ position: 'right', value: 'Threshold', fill: '#f59e0b', fontSize: 10 }} />
          
          <Area 
            type="monotone" 
            dataKey="fakeProbability" 
            stroke="#ef4444" 
            fillOpacity={1} 
            fill="url(#colorFake)" 
            strokeWidth={2}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};