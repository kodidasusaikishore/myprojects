import React from 'react';
import { motion } from 'framer-motion';
import { Typewriter } from 'react-simple-typewriter';
import { ArrowDown, Github, Linkedin, Mail } from 'lucide-react';

const Hero = () => {
  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative pt-16 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] bg-cyan-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse delay-700" />

      <div className="text-center z-10 max-w-4xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-cyan-400 font-mono text-lg mb-4 tracking-widest">HELLO, I AM</h2>
          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
            <span className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              Saikishore Kodidasu
            </span>
          </h1>
          
          <div className="text-2xl md:text-4xl font-bold text-gray-400 mb-8 h-[60px]">
            <span className="text-purple-400">
              <Typewriter
                words={['Assistant Vice President', 'Python Automation Architect', 'Data Driven Solver']}
                loop={0}
                cursor
                cursorStyle='_'
                typeSpeed={70}
                deleteSpeed={50}
                delaySpeed={1000}
              />
            </span>
          </div>

          <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
            Delivering Data Driven Solutions & Process Innovation in Financial Services.
            <br />
            Currently driving innovation at <span className="text-white font-semibold">Barclays</span>.
          </p>

          <div className="flex justify-center space-x-6 mb-12">
            <SocialLink href="https://github.com/kodidasusaikishore" icon={<Github />} />
            <SocialLink href="https://linkedin.com/in/saikishore-kodidasu-49198180" icon={<Linkedin />} />
            <SocialLink href="mailto:Saikishore.Kodidasu@gmail.com" icon={<Mail />} />
          </div>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1, duration: 1 }}
            className="animate-bounce"
          >
            <a href="#about" className="text-gray-500 hover:text-white transition-colors">
              <ArrowDown size={32} />
            </a>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

const SocialLink = ({ href, icon }) => (
  <a 
    href={href} 
    target="_blank" 
    rel="noopener noreferrer"
    className="p-3 border border-white/10 rounded-full hover:bg-white/10 hover:border-cyan-400/50 hover:text-cyan-400 transition-all duration-300"
  >
    {icon}
  </a>
);

export default Hero;
