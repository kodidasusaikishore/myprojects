import React from 'react';
import Section from './Section';
import { Briefcase } from 'lucide-react';

const Experience = () => {
  const experiences = [
    {
      role: "Assistant Vice President",
      company: "Barclays",
      period: "Jun 2021 - Present",
      description: "Leading automation initiatives, architecting Python-based solutions, and delivering data-driven process innovations in financial services."
    },
    {
      role: "FAR Automation Tester",
      company: "Barclays",
      period: "Aug 2018 - Jun 2021",
      description: "Specialized in Core Java and Test Automation, building robust frameworks for financial reporting validation."
    },
    {
      role: "Software Engineer",
      company: "Tech Mahindra",
      period: "Jun 2014 - Jul 2018",
      description: "Started as a Test Engineer handling both manual and automated testing processes, laying the foundation for advanced automation architecture."
    }
  ];

  return (
    <Section id="experience" title="Experience">
      <div className="space-y-12 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-300 before:to-transparent">
        {experiences.map((exp, index) => (
          <div key={index} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
            
            <div className="flex items-center justify-center w-10 h-10 rounded-full border border-white bg-slate-300 group-[.is-active]:bg-cyan-500 text-slate-500 group-[.is-active]:text-emerald-50 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2">
              <Briefcase size={20} />
            </div>
            
            <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-6 rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm shadow-lg hover:border-cyan-500/50 transition-colors duration-300">
              <div className="flex items-center justify-between space-x-2 mb-1">
                <div className="font-bold text-slate-200">{exp.role}</div>
                <time className="font-caveat font-medium text-cyan-400">{exp.period}</time>
              </div>
              <div className="text-slate-400 mb-2">{exp.company}</div>
              <div className="text-slate-400 text-sm">{exp.description}</div>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
};

export default Experience;
