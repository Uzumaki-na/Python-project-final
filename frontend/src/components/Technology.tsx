import { motion } from 'framer-motion';

const Technology = () => {
  return (
    <div className="bg-gray-800 py-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-white mb-4">Powered by Advanced Technology</h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Our platform leverages cutting-edge deep learning models and modern web technologies
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-gray-900/50 backdrop-blur-lg p-8 rounded-xl border border-gray-700"
          >
            <h3 className="text-2xl font-semibold text-white mb-6">Deep Learning Model</h3>
            <ul className="space-y-4">
              <li className="flex items-start">
                <svg className="w-6 h-6 text-emerald-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">EfficientNet-B0 Architecture</span>
                  <p className="text-gray-400 mt-1">State-of-the-art convolutional neural network optimized for medical image analysis</p>
                </div>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-emerald-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">PyTorch Framework</span>
                  <p className="text-gray-400 mt-1">Built on PyTorch for efficient training and inference</p>
                </div>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-emerald-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">Advanced Data Augmentation</span>
                  <p className="text-gray-400 mt-1">Robust training with sophisticated image augmentation techniques</p>
                </div>
              </li>
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-gray-900/50 backdrop-blur-lg p-8 rounded-xl border border-gray-700"
          >
            <h3 className="text-2xl font-semibold text-white mb-6">Modern Web Stack</h3>
            <ul className="space-y-4">
              <li className="flex items-start">
                <svg className="w-6 h-6 text-blue-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">React + TypeScript</span>
                  <p className="text-gray-400 mt-1">Type-safe frontend with modern React and Framer Motion animations</p>
                </div>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-blue-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">FastAPI Backend</span>
                  <p className="text-gray-400 mt-1">High-performance Python backend with automatic API documentation</p>
                </div>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-blue-400 mt-1 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                <div>
                  <span className="text-white font-medium">TailwindCSS</span>
                  <p className="text-gray-400 mt-1">Utility-first CSS framework for beautiful, responsive design</p>
                </div>
              </li>
            </ul>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Technology;
