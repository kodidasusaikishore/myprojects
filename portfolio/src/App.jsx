import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import About from './components/About';
import Experience from './components/Experience';
import Certifications from './components/Certifications';
import Projects from './components/Projects';
import Hobbies from './components/Hobbies';
import Contact from './components/Contact';

function App() {
  return (
    <div className="bg-dark-bg text-white min-h-screen selection:bg-neon-blue selection:text-black">
      <Navbar />
      <main className="container mx-auto px-4 sm:px-6 lg:px-8">
        <Hero />
        <About />
        <Experience />
        <Certifications />
        <Hobbies />
        <Projects />
        <Contact />
      </main>
      
      <footer className="py-8 text-center text-gray-500 text-sm">
        <p>© {new Date().getFullYear()} Saikishore Kodidasu. Built with React & Tailwind.</p>
      </footer>
    </div>
  );
}

export default App;
