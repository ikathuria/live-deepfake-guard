import React from 'react';

interface MetricsCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  colorClass?: string;
  icon?: React.ReactNode;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({ 
  title, 
  value, 
  subtext, 
  colorClass = "text-white",
  icon 
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg flex flex-col justify-between h-32 relative overflow-hidden group">
      <div className="flex justify-between items-start z-10">
        <h3 className="text-slate-400 text-xs font-semibold uppercase tracking-wider">{title}</h3>
        {icon && <div className="text-slate-600 group-hover:text-slate-400 transition-colors">{icon}</div>}
      </div>
      <div className="z-10 mt-2">
        <p className={`text-3xl font-bold ${colorClass}`}>{value}</p>
        {subtext && <p className="text-slate-500 text-xs mt-1">{subtext}</p>}
      </div>
      
      {/* Abstract bg decoration */}
      <div className="absolute -bottom-4 -right-4 w-24 h-24 bg-slate-800 rounded-full opacity-10 group-hover:scale-150 transition-transform duration-700 pointer-events-none"></div>
    </div>
  );
};