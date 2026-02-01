import React from 'react';
import Section from './Section';
import { Award, CheckCircle } from 'lucide-react';

const Certifications = () => {
  const certs = [
    {
      name: "AWS Certified Cloud Practitioner",
      issuer: "Amazon Web Services (AWS)",
      date: "Issued Jan 2026 · Expires Jan 2029",
      skills: "Cloud Computing · AWS"
    },
    {
      name: "Python Certification",
      issuer: "LearnTube.ai",
      date: "Issued Jul 2024",
      skills: "Python (Programming Language)"
    },
    {
      name: "ISTQB Foundation Level Certification",
      issuer: "ISTQB - International Software Testing Qualifications Board",
      date: "Issued Dec 2015",
      skills: "Software Testing · QA"
    }
  ];

  return (
    <Section id="certifications" title="Licenses & Certifications">
      <div className="grid md:grid-cols-3 gap-6">
        {certs.map((cert, index) => (
          <div key={index} className="bg-white/5 border border-white/10 p-6 rounded-xl hover:bg-white/10 transition-colors duration-300 flex flex-col h-full">
            <div className="mb-4 text-cyan-400">
              <Award size={40} />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">{cert.name}</h3>
            <p className="text-gray-400 text-sm mb-1">{cert.issuer}</p>
            <p className="text-gray-500 text-xs mb-4">{cert.date}</p>
            
            <div className="mt-auto pt-4 border-t border-white/5 flex items-center text-sm text-cyan-300">
              <CheckCircle size={14} className="mr-2" />
              <span>{cert.skills}</span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
};

export default Certifications;
