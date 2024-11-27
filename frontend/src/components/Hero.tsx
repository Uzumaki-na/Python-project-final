import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';

const Hero = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/detect');
  };

  const technologies = [
    { name: 'React', description: 'Frontend Framework' },
    { name: 'FastAPI', description: 'Backend API' },
    { name: 'PyTorch', description: 'Deep Learning' },
    { name: 'EfficientNet', description: 'CNN Architecture' },
    { name: 'TypeScript', description: 'Type Safety' },
    { name: 'TailwindCSS', description: 'Styling' },
  ];

  const metrics = [
    { label: 'Skin Cancer Detection', value: '97%', description: 'Accuracy on test set' },
    { label: 'Malaria Detection', value: '96%', description: 'Accuracy on test set' },
    { label: 'Processing Time', value: '<2s', description: 'Per image' },
    { label: 'Model Size', value: '11MB', description: 'Optimized for web' },
  ];

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Hero Section */}
      <div className="relative pt-16 pb-32">
        <div className="absolute inset-0 bg-grid-pattern opacity-5" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight"
            >
              <span className="block text-white">AI-Powered Medical</span>
              <span className="block text-blue-500">Image Analysis</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mt-6 text-lg sm:text-xl text-gray-400 max-w-3xl mx-auto"
            >
              Advanced machine learning technology for rapid and accurate detection of skin cancer and malaria.
              Upload medical images and get instant analysis with high confidence predictions.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mt-8 flex flex-wrap justify-center gap-4"
            >
              <button
                onClick={handleGetStarted}
                className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-500 hover:bg-blue-600 transition-all duration-300"
              >
                Get Started
              </button>
              <a
                href="https://github.com/yourusername/medical-image-analysis"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-blue-500 bg-gray-800 hover:bg-gray-700 transition-all duration-300"
              >
                View on GitHub
              </a>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div className="py-16 bg-gray-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4"
          >
            {metrics.map((metric, index) => (
              <div key={index} className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
                <div className="relative p-6 bg-gray-800 rounded-lg text-center">
                  <p className="text-2xl font-bold text-white">{metric.value}</p>
                  <p className="text-sm font-medium text-blue-400">{metric.label}</p>
                  <p className="text-xs text-gray-400 mt-1">{metric.description}</p>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-3xl font-bold text-white mb-4"
            >
              Advanced Medical Image Analysis
            </motion.h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Our platform leverages state-of-the-art deep learning models to provide accurate and reliable medical image analysis.
            </p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8"
          >
            {/* Feature 1 */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
              <div className="relative p-6 bg-gray-800 ring-1 ring-gray-700/5 rounded-lg h-full">
                <h3 className="text-xl font-semibold text-white mb-3">Skin Cancer Detection</h3>
                <p className="text-gray-400">
                  Upload images of skin lesions for instant analysis. Our AI model detects potential melanoma and other skin cancers with high accuracy.
                </p>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
              <div className="relative p-6 bg-gray-800 ring-1 ring-gray-700/5 rounded-lg h-full">
                <h3 className="text-xl font-semibold text-white mb-3">Malaria Detection</h3>
                <p className="text-gray-400">
                  Analyze blood smear images to detect malaria parasites. Quick and reliable diagnosis to support healthcare professionals.
                </p>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="relative group sm:col-span-2 lg:col-span-1">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
              <div className="relative p-6 bg-gray-800 ring-1 ring-gray-700/5 rounded-lg h-full">
                <h3 className="text-xl font-semibold text-white mb-3">Advanced AI Technology</h3>
                <p className="text-gray-400">
                  Powered by state-of-the-art deep learning models. Fast, accurate, and continuously improving through machine learning.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="py-24 bg-gray-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-3xl font-bold text-white mb-4"
            >
              Technology Stack
            </motion.h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Built with modern technologies for performance, reliability, and scalability
            </p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-8"
          >
            {technologies.map((tech, index) => (
              <div key={index} className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200" />
                <div className="relative p-4 bg-gray-800 rounded-lg text-center">
                  <h3 className="text-lg font-semibold text-white">{tech.name}</h3>
                  <p className="text-sm text-gray-400">{tech.description}</p>
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-3xl font-bold text-white mb-6"
            >
              Ready to Get Started?
            </motion.h2>
            <p className="text-gray-400 max-w-2xl mx-auto mb-8">
              Experience the power of AI in medical image analysis. Upload your first image and get instant results.
            </p>
            <button
              onClick={handleGetStarted}
              className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-500 hover:bg-blue-600 transition-all duration-300"
            >
              Try It Now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
