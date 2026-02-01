import React from 'react';
import { motion } from 'framer-motion';

const Section = ({ id, title, children }) => (
  <section id={id} className="py-20 border-t border-white/5">
    <div className="max-w-6xl mx-auto px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">
          <span className="border-b-2 border-cyan-400 pb-2">{title}</span>
        </h2>
        {children}
      </motion.div>
    </div>
  </section>
);

export default Section;
