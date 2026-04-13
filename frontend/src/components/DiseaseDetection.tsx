import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import config from '../config';

interface DetectionResult {
    message: string;
    prediction: string | null;
    confidence: number;
}

const ImageUpload: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<DetectionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [selectedTest, setSelectedTest] = useState<'skin-cancer' | 'malaria'>('skin-cancer');
    const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

    useEffect(() => {
        checkBackendStatus();
    }, []);

    const checkBackendStatus = async () => {
        try {
            await axios.get(`${config.API_URL}/health`);
            setBackendStatus('online');
            setError(null);
        } catch (err) {
            console.error('Backend health check failed:', err);
            setBackendStatus('offline');
            setError('Cannot connect to backend server');
        }
    };

    const validateFile = (file: File): string | null => {
        if (!config.SUPPORTED_IMAGE_TYPES.includes(file.type)) {
            return 'Please upload a valid image file (JPEG, PNG, or GIF)';
        }
        if (file.size > config.MAX_IMAGE_SIZE) {
            return 'Image size should be less than 5MB';
        }
        return null;
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const validationError = validateFile(file);
            if (validationError) {
                setError(validationError);
                return;
            }
            
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
            
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setResult(null);
            setError(null);
        }
    };

    const handleSubmit = async () => {
        if (backendStatus !== 'online') {
            setError('Backend server is not available. Please try again later.');
            return;
        }

        if (!selectedFile) {
            setError('Please select an image first');
            return;
        }

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const endpoint = selectedTest === 'skin-cancer' ? config.ENDPOINTS.SKIN_CANCER : config.ENDPOINTS.MALARIA;
            const response = await axios.post(`${config.API_URL}${endpoint}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResult(response.data);
        } catch (err) {
            console.error('Error:', err);
            setError(err instanceof Error ? err.message : 'Error processing image. Please try again.');
            // If we get a connection error, recheck backend status
            if (axios.isAxiosError(err) && !err.response) {
                checkBackendStatus();
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-8">
            {backendStatus === 'checking' && (
                <div className="mb-6 p-4 bg-blue-50 text-blue-700 rounded-lg">
                    <div className="flex items-center">
                        <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Checking server status...
                    </div>
                </div>
            )}
            
            {backendStatus === 'offline' && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg">
                    <div className="flex items-center">
                        <svg className="h-5 w-5 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Server is offline. Please try again later.
                    </div>
                </div>
            )}

            <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">Select Test Type</h2>
                <div className="grid grid-cols-2 gap-4">
                    <button
                        onClick={() => setSelectedTest('skin-cancer')}
                        className={`p-4 rounded-lg transition-all duration-200 ${
                            selectedTest === 'skin-cancer'
                                ? 'bg-blue-600 text-white shadow-md'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        } disabled:opacity-50`}
                        disabled={backendStatus !== 'online'}
                    >
                        <div className="text-lg font-medium">Skin Cancer</div>
                        <div className="text-sm mt-1 opacity-75">Melanoma Detection</div>
                    </button>
                    <button
                        onClick={() => setSelectedTest('malaria')}
                        className={`p-4 rounded-lg transition-all duration-200 ${
                            selectedTest === 'malaria'
                                ? 'bg-blue-600 text-white shadow-md'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        } disabled:opacity-50`}
                        disabled={backendStatus !== 'online'}
                    >
                        <div className="text-lg font-medium">Malaria</div>
                        <div className="text-sm mt-1 opacity-75">Parasite Detection</div>
                    </button>
                </div>
            </div>

            <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
                <div className="mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-blue-500 transition-colors">
                    <div className="space-y-1 text-center">
                        <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        <div className="flex text-sm text-gray-600">
                            <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
                                <span>Upload a file</span>
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileSelect}
                                    disabled={backendStatus !== 'online'}
                                    className="sr-only"
                                />
                            </label>
                            <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">
                            Supported: JPEG, PNG, GIF (Max: 5MB)
                        </p>
                    </div>
                </div>
            </div>

            {previewUrl && (
                <div className="mb-8">
                    <h2 className="text-xl font-semibold mb-4">Image Preview</h2>
                    <div className="rounded-lg overflow-hidden shadow-lg">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="w-full h-auto"
                        />
                    </div>
                </div>
            )}

            {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg">
                    <div className="flex items-center">
                        <svg className="h-5 w-5 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {error}
                    </div>
                </div>
            )}

            <button
                onClick={handleSubmit}
                disabled={loading || !selectedFile || backendStatus !== 'online'}
                className="w-full py-3 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center justify-center"
            >
                {loading ? (
                    <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                    </>
                ) : (
                    'Analyze Image'
                )}
            </button>

            {result && (
                <div className="mt-8 bg-gray-50 rounded-lg p-6">
                    <h2 className="text-xl font-semibold mb-4">Results</h2>
                    {result.message && (
                        <p className="text-gray-600 mb-4">{result.message}</p>
                    )}
                    {result.prediction && (
                        <div className="space-y-4">
                            <div className="bg-white rounded-lg p-4 shadow">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <p className="text-sm text-gray-500">Prediction</p>
                                        <p className="text-lg font-semibold text-gray-800">{result.prediction}</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm text-gray-500">Confidence</p>
                                        <p className="text-lg font-semibold text-blue-600">{(result.confidence * 100).toFixed(2)}%</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ImageUpload;
