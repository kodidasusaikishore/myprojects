import React from 'react';
import Section from './Section';

const About = () => {
  return (
    <Section id="about" title="About Me">
      <div className="grid md:grid-cols-2 gap-10 items-center">
        <div className="bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-sm">
           {/* Profile Image */}
           <div className="aspect-square rounded-xl overflow-hidden border-2 border-cyan-400/30 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
             <img 
               src={`${import.meta.env.BASE_URL}profile-formal.jpg`} 
               alt="Saikishore Kodidasu" 
               className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
             />
           </div>
        </div>
        <div className="space-y-6 text-gray-300 text-lg leading-relaxed">
          <p>
            In a world driven by data, its quality, accuracy, and speed are non-negotiable. 
            I specialize in ensuring that the data powering your business is flawless and reliable.
          </p>
          <p>
            As a <span className="text-cyan-400 font-semibold">Lead Automation Test Engineer</span> and Architect, 
            I go beyond traditional testing. I architect and build custom automation frameworks from the ground up 
            to solve complex data validation challenges. Where others see manual bottlenecks, I see an opportunity 
            to create elegant, efficient solutions.
          </p>
          <p>
            My toolkit is versatile, leveraging languages like <span className="text-cyan-400">Python, Java, and Shell</span> to 
            build robust tools that interface with a wide range of databases—from SQL Server and MySQL to Hive and Coherence.
          </p>
          <div className="flex flex-wrap gap-3 mt-6">
            {['Python', 'Java', 'Shell Scripting', 'SQL Server', 'MySQL', 'Hive', 'Coherence', 'Automation Architecture'].map((tag) => (
              <span key={tag} className="px-4 py-2 bg-cyan-900/30 text-cyan-300 rounded-full text-sm border border-cyan-800/50">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
};

export default About;
