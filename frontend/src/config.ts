const config = {
  API_URL: 'http://127.0.0.1:8000',
  ENDPOINTS: {
    SKIN_CANCER: '/predict/skin-cancer',
    MALARIA: '/predict/malaria',
  },
  MAX_IMAGE_SIZE: 5 * 1024 * 1024, // 5MB
  SUPPORTED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/gif'],
};

export default config;
