import { useState, useRef } from 'react';
import { motion } from 'framer-motion';

type DiseaseType = 'skin-cancer' | 'malaria';

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diseaseType, setDiseaseType] = useState<DiseaseType>('skin-cancer');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch(`http://localhost:8000/predict/${diseaseType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError('Error analyzing image. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-12">
      <div className="max-w-3xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-gray-800 rounded-xl shadow-xl overflow-hidden"
        >
          <div className="p-6">
            <h2 className="text-2xl font-bold text-white mb-4 text-center">Medical Image Analysis</h2>
            <p className="text-gray-400 text-center text-sm mb-6">Select disease type and upload an image for analysis</p>
            
            {/* Disease Type Selection */}
            <div className="flex justify-center gap-4 mb-6">
              <button
                onClick={() => setDiseaseType('skin-cancer')}
                className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  diseaseType === 'skin-cancer'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Skin Cancer Detection
              </button>
              <button
                onClick={() => setDiseaseType('malaria')}
                className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  diseaseType === 'malaria'
                    ? 'bg-emerald-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Malaria Detection
              </button>
            </div>
            
            {/* Upload Area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors duration-300"
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageChange}
                accept="image/*"
                className="hidden"
              />
              
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded-lg"
                />
              ) : (
                <div className="text-gray-400">
                  <svg className="mx-auto h-12 w-12 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-base">Drop an image here or click to upload</p>
                  <p className="text-xs mt-2">Supports: JPG, PNG, GIF</p>
                </div>
              )}
            </div>

            {/* Analysis Button */}
            {selectedImage && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="mt-6 text-center"
              >
                <button
                  onClick={analyzeImage}
                  disabled={loading}
                  className={`px-8 py-3 rounded-lg text-base font-semibold disabled:opacity-50 hover:shadow-lg transition-all duration-300 ${
                    diseaseType === 'skin-cancer'
                      ? 'bg-gradient-to-r from-blue-500 to-blue-600'
                      : 'bg-gradient-to-r from-emerald-500 to-emerald-600'
                  } text-white`}
                >
                  {loading ? 'Analyzing...' : `Classify ${diseaseType === 'skin-cancer' ? 'Skin Lesion' : 'Blood Cell'}`}
                </button>
              </motion.div>
            )}

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm text-center"
              >
                {error}
              </motion.div>
            )}

            {/* Results */}
            {prediction && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-6 bg-gray-700/50 rounded-lg"
              >
                <h3 className="text-lg font-semibold text-white mb-3">Analysis Results</h3>
                <div className="space-y-2">
                  <p className="text-gray-300">
                    <span className="font-medium">Prediction:</span>{' '}
                    <span className={
                      diseaseType === 'skin-cancer'
                        ? prediction.prediction === 'malignant' ? 'text-red-400' : 'text-green-400'
                        : prediction.prediction === 'parasitized' ? 'text-red-400' : 'text-green-400'
                    }>
                      {prediction.prediction}
                    </span>
                  </p>
                  <p className="text-gray-300">
                    <span className="font-medium">Confidence:</span>{' '}
                    {(prediction.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ImageUpload;
