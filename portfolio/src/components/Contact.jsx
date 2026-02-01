import React from 'react';
import Section from './Section';
import { Mail, MapPin, Linkedin } from 'lucide-react';

const Contact = () => {
  return (
    <Section id="contact" title="Get In Touch">
      <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-10">
        <div className="space-y-8">
          <p className="text-gray-300 text-lg">
            I am always open to discussing new projects, creative ideas or opportunities to be part of your visions.
          </p>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-4 text-gray-400">
              <div className="bg-white/10 p-3 rounded-full">
                <MapPin className="text-cyan-400" />
              </div>
              <span>Pune, Maharashtra, India</span>
            </div>
            <div className="flex items-center space-x-4 text-gray-400">
              <div className="bg-white/10 p-3 rounded-full">
                <Mail className="text-cyan-400" />
              </div>
              <a href="mailto:Saikishore.Kodidasu@gmail.com" className="hover:text-white transition-colors">
                Saikishore.Kodidasu@gmail.com
              </a>
            </div>
             <div className="flex items-center space-x-4 text-gray-400">
              <div className="bg-white/10 p-3 rounded-full">
                <Linkedin className="text-cyan-400" />
              </div>
              <a href="https://linkedin.com/in/saikishore-kodidasu-49198180" target="_blank" className="hover:text-white transition-colors">
                LinkedIn Profile
              </a>
            </div>
          </div>
        </div>

        <form className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <input type="text" placeholder="Name" className="bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-400 transition-colors" />
            <input type="email" placeholder="Email" className="bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-400 transition-colors" />
          </div>
          <input type="text" placeholder="Subject" className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-400 transition-colors" />
          <textarea placeholder="Message" rows="4" className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-400 transition-colors"></textarea>
          <button className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold py-3 px-8 rounded-lg hover:shadow-lg hover:shadow-cyan-500/20 transition-all transform hover:-translate-y-1">
            Send Message
          </button>
        </form>
      </div>
    </Section>
  );
};

export default Contact;
