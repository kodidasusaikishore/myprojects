import React from 'react';
import Section from './Section';
import { Camera, Heart, Globe } from 'lucide-react';

const Hobbies = () => {
  return (
    <Section id="hobbies" title="Beyond the Code">
      <div className="grid md:grid-cols-2 gap-12 items-center">
        
        {/* Content Side */}
        <div className="order-2 md:order-1 space-y-6">
          <h3 className="text-2xl font-bold text-white flex items-center">
            <Camera className="text-cyan-400 mr-3" size={28} />
            Photography & Exploration
          </h3>
          
          <p className="text-gray-300 text-lg leading-relaxed">
            When I step away from automation architecture, I step behind the lens. 
            <span className="text-cyan-400"> Photography</span> is my way of capturing the world's stillness amidst the chaos. 
            Just as I look for patterns in data, I look for patterns in light and shadow.
          </p>
          
          <p className="text-gray-300 text-lg leading-relaxed">
            I am also an avid traveler. Exploring new landscapes and cultures fuels my creativity 
            and offers fresh perspectives that I bring back to my problem-solving approach.
          </p>

          <div className="flex gap-4 pt-4">
             <div className="flex items-center text-sm text-gray-400 bg-white/5 px-4 py-2 rounded-full border border-white/10">
                <Heart size={16} className="text-pink-500 mr-2" /> Traveler
             </div>
             <div className="flex items-center text-sm text-gray-400 bg-white/5 px-4 py-2 rounded-full border border-white/10">
                <Globe size={16} className="text-blue-500 mr-2" /> Explorer
             </div>
          </div>
        </div>

        {/* Image Side */}
        <div className="order-1 md:order-2 relative">
           <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-2xl blur-lg opacity-30 animate-pulse"></div>
           <div className="relative aspect-[3/4] md:aspect-[4/3] rounded-2xl overflow-hidden border border-white/10 shadow-2xl rotate-2 hover:rotate-0 transition-all duration-500">
             <img 
               src={`${import.meta.env.BASE_URL}hobby-camera.jpg`} 
               alt="Photography Hobby" 
               className="w-full h-full object-cover"
             />
             <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6">
                <p className="text-white font-caveat text-xl">"Capturing moments, one frame at a time."</p>
             </div>
           </div>
        </div>

      </div>
    </Section>
  );
};

export default Hobbies;
