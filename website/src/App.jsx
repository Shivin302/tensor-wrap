import React from 'react';
import { useState, useEffect } from 'react';

export default function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('home');

  // Handle scroll to detect active section
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['home', 'problem', 'solution', 'team', 'contact'];
      const scrollPosition = window.scrollY + 100;

      for (const section of sections) {
        const element = document.getElementById(section);
        if (
          element &&
          element.offsetTop <= scrollPosition &&
          element.offsetTop + element.offsetHeight > scrollPosition
        ) {
          setActiveSection(section);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="bg-white text-gray-800 font-sans">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white shadow-sm backdrop-blur-md bg-opacity-90 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex-shrink-0 flex items-center">
              <span className="text-xl font-bold text-indigo-600">AI Agent</span>
            </div>

            {/* Desktop Menu */}
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                {[
                  { name: 'Home', id: 'home' },
                  { name: 'Problem', id: 'problem' },
                  { name: 'Solution', id: 'solution' },
                  { name: 'Team', id: 'team' },
                  { name: 'Contact', id: 'contact' },
                ].map((item) => (
                  <a
                    key={item.id}
                    href={`#${item.id}`}
                    className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${activeSection === item.id
                        ? 'text-indigo-600 border-b-2 border-indigo-600'
                        : 'text-gray-700 hover:text-indigo-600'
                      }`}
                  >
                    {item.name}
                  </a>
                ))}
              </div>
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-indigo-600 focus:outline-none"
              >
                <svg
                  className={`${isMenuOpen ? 'hidden' : 'block'} h-6 w-6`}
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                <svg
                  className={`${isMenuOpen ? 'block' : 'hidden'} h-6 w-6`}
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <div className={`${isMenuOpen ? 'block' : 'hidden'} md:hidden bg-white`}>
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {[
              { name: 'Home', id: 'home' },
              { name: 'Problem', id: 'problem' },
              { name: 'Solution', id: 'solution' },
              { name: 'Team', id: 'team' },
              { name: 'Contact', id: 'contact' },
            ].map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${activeSection === item.id
                    ? 'text-indigo-600 bg-indigo-50'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-indigo-600'
                  }`}
                onClick={() => setIsMenuOpen(false)}
              >
                {item.name}
              </a>
            ))}
          </div>
        </div>
      </nav>

      <main>
        {/* Hero Section */}
        <section id="home" className="pt-24 pb-16 md:py-32 bg-gradient-to-br from-white via-indigo-50 to-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <h1 className="text-4xl md:text-5xl font-extrabold leading-tight text-gray-900">
                  Automating ML Infrastructure at Scale
                </h1>
                <p className="text-lg text-gray-600">
                  AI Agents that reduce cloud costs, speed up inference, and let one engineer do the work of five.
                </p>
                <div className="flex flex-col sm:flex-row gap-4">
                  <a
                    href="#contact"
                    className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition-colors"
                  >
                    Request Early Access
                  </a>
                  <a
                    href="#solution"
                    className="px-6 py-3 border border-indigo-600 text-indigo-600 font-semibold rounded-lg hover:bg-indigo-50 transition-colors"
                  >
                    Learn More
                  </a>
                </div>
              </div>
              <div className="relative">
                <div className="absolute inset-0 bg-indigo-100 rounded-xl transform rotate-3 scale-105"></div>
                <img
                  src="https://placehold.co/600x400/indigo/white?text=AI+Agents+at+Work "
                  alt="AI Agent in action"
                  className="relative rounded-xl shadow-lg"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Problem Section */}
        <section id="problem" className="py-20 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-gray-900">The Problem</h2>
              <p className="mt-4 max-w-2xl mx-auto text-lg text-gray-600">
                Scaling AI infrastructure is complex, expensive, and time-consuming.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  title: 'High Latency',
                  description:
                    'Every millisecond saved in inference leads to faster user experience and millions saved annually.',
                },
                {
                  title: 'Costly Engineering',
                  description:
                    'Hiring full teams of low-level engineers and DevOps specialists drains resources and slows innovation.',
                },
                {
                  title: 'Manual Optimization',
                  description:
                    'From CUDA kernels to model quantization, optimization remains a tedious manual process.',
                },
              ].map((card, index) => (
                <div
                  key={index}
                  className="p-6 bg-gray-50 rounded-xl shadow-sm hover:shadow-md transition-shadow border border-gray-100"
                >
                  <h3 className="text-xl font-semibold text-gray-800">{card.title}</h3>
                  <p className="mt-3 text-gray-600">{card.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Solution Section */}
        <section id="solution" className="py-20 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-gray-900">Our Solution</h2>
              <p className="mt-4 max-w-2xl mx-auto text-lg text-gray-600">
                AI Agents that automate your entire ML pipeline from training to deployment.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-gray-900">Intelligent Automation</h3>
                <ul className="space-y-4 text-gray-600">
                  <li className="flex items-start">
                    <span className="mr-2 text-green-500">&#x2022;</span>
                    <span>Auto-scale your training and inference pipelines across GPUs and TPUs.</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-green-500">&#x2022;</span>
                    <span>Automate model compression, quantization, and kernel optimization.</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-green-500">&#x2022;</span>
                    <span>Reduce latency by automatically choosing optimal hardware configurations.</span>
                  </li>
                  <li className="flex items-start">
                    <span className="mr-2 text-green-500">&#x2022;</span>
                    <span>Self-learning agents adapt to new models and infrastructures over time.</span>
                  </li>
                </ul>
              </div>
              <div className="relative">
                <div className="absolute inset-0 bg-indigo-100 rounded-xl transform -rotate-2 scale-105"></div>
                <img
                  src="https://placehold.co/600x400/indigo/white?text=Agent+Dashboard "
                  alt="Agent Dashboard"
                  className="relative rounded-xl shadow-lg"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section id="team" className="py-20 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-gray-900">Founding Team</h2>
              <p className="mt-4 max-w-2xl mx-auto text-lg text-gray-600">
                Experience at Alphabet, Apple, and startups building scalable AI systems.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              {[
                {
                  name: 'Saurav Mittal',
                  role: 'CEO & Co-founder',
                  bio: 'Led core ML infrastructure for Apple Intelligence and Apple Special Project Group',
                  image: 'https://placehold.co/200x200/indigo/white?text=Alex ',
                },
                {
                  name: 'Shivin Devgon',
                  role: 'CTO & Co-founder',
                  bio: 'Led model optimization at Alphabet for Waymo\'s Vision Transformers.',
                  image: 'https://placehold.co/200x200/indigo/white?text=Jamie ',
                },
              ].map((member, index) => (
                <div
                  key={index}
                  className="flex items-start space-x-6 bg-gray-50 p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                >
                  <img
                    src={member.image}
                    alt={member.name}
                    className="w-20 h-20 rounded-full object-cover"
                  />
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">{member.name}</h3>
                    <p className="text-indigo-600 font-medium">{member.role}</p>
                    <p className="mt-2 text-gray-600">{member.bio}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="py-20 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-gray-900">Get in Touch</h2>
              <p className="mt-4 max-w-2xl mx-auto text-lg text-gray-600">
                We're offering early access to select companies. Let's talk about how we can help you scale.
              </p>
            </div>

            <div className="max-w-3xl mx-auto">
              <form className="bg-white shadow-md rounded-xl p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                      Full Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                      Email Address
                    </label>
                    <input
                      type="email"
                      id="email"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
                      placeholder="you@example.com"
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">
                    Message
                  </label>
                  <textarea
                    id="message"
                    rows="4"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
                    placeholder="Tell us about your use case..."
                  ></textarea>
                </div>
                <button
                  type="submit"
                  className="w-full px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition-colors"
                >
                  Send Message
                </button>
              </form>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <span className="text-lg font-bold text-indigo-600">AI Agent</span>
            </div>
            <div className="text-sm text-gray-600">
              &copy; {new Date().getFullYear()} AI Agent Inc. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}