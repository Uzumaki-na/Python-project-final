import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Hero from './components/Hero';
import ImageUpload from './components/ImageUpload';
import './App.css';

const App = () => {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<Hero />} />
          <Route path="/detect" element={<ImageUpload />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
