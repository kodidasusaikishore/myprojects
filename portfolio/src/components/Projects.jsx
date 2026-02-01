import React from 'react';
import Section from './Section';
import { ExternalLink, Github } from 'lucide-react';

const Projects = () => {
  const projects = [
    {
      title: "Portfolio Website",
      desc: "A futuristic, interactive portfolio built with React, Tailwind, and Framer Motion.",
      tech: ["React", "Tailwind", "Framer Motion"],
      link: "https://github.com/kodidasusaikishore/myprojects/tree/main/portfolio"
    },
    {
      title: "PreciousPulse 💎",
      desc: "AI-powered bullion tracker & predictor using LSTM & Sentiment Analysis.",
      tech: ["Streamlit", "Python", "TensorFlow", "Plotly"],
      link: "https://stock-prediction-lstm.streamlit.app/"
    },
    {
      title: "Stock Price Prediction",
      desc: "Predicting stock prices using LSTM neural networks with interactive charts.",
      tech: ["Streamlit", "Python", "TensorFlow", "yfinance"],
      link: "https://stock-prediction-lstm.streamlit.app/"
    }
    // Add more later
  ];

  return (
    <Section id="projects" title="Featured Projects">
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
        {projects.map((project, index) => (
          <div key={index} className="group relative bg-white/5 border border-white/10 rounded-xl overflow-hidden hover:-translate-y-2 transition-transform duration-300">
            <div className="h-48 bg-gradient-to-br from-gray-800 to-black flex items-center justify-center">
                <span className="text-4xl">🚀</span>
            </div>
            <div className="p-6">
              <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-400 transition-colors">{project.title}</h3>
              <p className="text-gray-400 mb-4">{project.desc}</p>
              <div className="flex flex-wrap gap-2 mb-6">
                {project.tech.map(t => (
                  <span key={t} className="text-xs px-2 py-1 bg-white/10 rounded text-gray-300">{t}</span>
                ))}
              </div>
              <div className="flex gap-4">
                <a href={project.link} className="flex items-center text-sm text-cyan-400 hover:underline">
                  <Github size={16} className="mr-1" /> Code
                </a>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
};

export default Projects;
