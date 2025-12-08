<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science, ML & DL Repository</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            overflow-x: hidden;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 60px 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: fadeInDown 1s ease;
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .badges {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin: 20px 0;
        }

        .badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            transition: transform 0.3s;
        }

        .badge:hover {
            transform: translateY(-5px);
        }

        .badge-python { background: #3776AB; color: white; }
        .badge-jupyter { background: #F37626; color: white; }
        .badge-ml { background: #4CAF50; color: white; }
        .badge-dl { background: #E91E63; color: white; }

        .tagline {
            font-size: 1.3em;
            color: #555;
            margin: 20px 0;
            font-style: italic;
        }

        .nav-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 30px;
        }

        .nav-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .nav-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .content-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h2 {
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .feature-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }

        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            border-color: #667eea;
        }

        .feature-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.3em;
        }

        .feature-card ul {
            list-style: none;
            padding-left: 0;
        }

        .feature-card li {
            padding: 5px 0;
            color: #555;
        }

        .feature-card li::before {
            content: "✓ ";
            color: #4CAF50;
            font-weight: bold;
            margin-right: 5px;
        }

        .accordion {
            background: #f5f7fa;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
            transition: all 0.3s;
        }

        .accordion-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1.2em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
        }

        .accordion-header:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

        .accordion-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            padding: 0 20px;
        }

        .accordion.active .accordion-content {
            max-height: 500px;
            padding: 20px;
        }

        .accordion-arrow {
            transition: transform 0.3s;
        }

        .accordion.active .accordion-arrow {
            transform: rotate(180deg);
        }

        .highlight-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .highlight-box h3 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }

        .stat-number {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            color: #666;
            margin-top: 10px;
            font-size: 1.1em;
        }

        code {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }

        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 20px 0;
        }

        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin: 30px 0;
        }

        .tech-badge {
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: bold;
            color: white;
            transition: all 0.3s;
            cursor: pointer;
        }

        .tech-badge:hover {
            transform: scale(1.1) rotate(5deg);
        }

        .tech-python { background: #3776AB; }
        .tech-numpy { background: #013243; }
        .tech-pandas { background: #150458; }
        .tech-sklearn { background: #F7931E; }
        .tech-tensorflow { background: #FF6F00; }
        .tech-pytorch { background: #EE4C2C; }

        .cta-section {
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            color: white;
            margin: 30px 0;
        }

        .cta-button {
            padding: 15px 40px;
            background: white;
            color: #667eea;
            border: none;
            border-radius: 30px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s;
        }

        .cta-button:hover {
            transform: scale(1.1);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .floating-emoji {
            position: fixed;
            font-size: 3em;
            animation: float 3s ease-in-out infinite;
            pointer-events: none;
            z-index: 1000;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }

        footer {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin-top: 30px;
        }

        .social-links {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }

        .social-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.5em;
            transition: all 0.3s;
            cursor: pointer;
        }

        .social-btn:hover {
            transform: scale(1.2) rotate(360deg);
        }

        .github { background: #333; }
        .linkedin { background: #0077B5; }
        .twitter { background: #1DA1F2; }

        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .feature-grid { grid-template-columns: 1fr; }
            .nav-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="floating-emoji" style="left: 10%; top: 10%;">🚀</div>
    <div class="floating-emoji" style="right: 10%; top: 20%; animation-delay: 1s;">📊</div>
    <div class="floating-emoji" style="left: 15%; bottom: 20%; animation-delay: 2s;">🤖</div>
    <div class="floating-emoji" style="right: 15%; bottom: 15%; animation-delay: 1.5s;">🧠</div>

    <div class="container">
        <header>
            <h1>🚀 Data Science, ML & Deep Learning</h1>
            <p class="tagline">Transform Data into Insights, Models into Predictions</p>
            
            <div class="badges">
                <span class="badge badge-python">Python 3.8+</span>
                <span class="badge badge-jupyter">Jupyter Notebook</span>
                <span class="badge badge-ml">Machine Learning</span>
                <span class="badge badge-dl">Deep Learning</span>
            </div>

            <div class="nav-buttons">
                <button class="nav-btn" onclick="scrollToSection('features')">📚 Explore</button>
                <button class="nav-btn" onclick="scrollToSection('quickstart')">🎯 Quick Start</button>
                <button class="nav-btn" onclick="scrollToSection('projects')">💡 Projects</button>
                <button class="nav-btn" onclick="scrollToSection('contact')">🤝 Connect</button>
            </div>
        </header>

        <div class="content-section">
            <h2>🎯 Why This Repository?</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">50+</div>
                    <div class="stat-label">Notebooks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">10+</div>
                    <div class="stat-label">ML Algorithms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">15+</div>
                    <div class="stat-label">Projects</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">100%</div>
                    <div class="stat-label">Open Source</div>
                </div>
            </div>
        </div>

        <div class="content-section" id="features">
            <h2>📚 What's Inside</h2>
            <div class="feature-grid">
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Fundamentals</h3>
                    <ul>
                        <li>Python Basics</li>
                        <li>NumPy & Pandas</li>
                        <li>Data Structures</li>
                        <li>Data Preprocessing</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 EDA</h3>
                    <ul>
                        <li>Data Visualization</li>
                        <li>Statistical Analysis</li>
                        <li>Feature Engineering</li>
                        <li>Data Profiling</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Classical ML</h3>
                    <ul>
                        <li>Linear/Logistic Regression</li>
                        <li>Decision Trees</li>
                        <li>Random Forests</li>
                        <li>XGBoost & LightGBM</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Unsupervised Learning</h3>
                    <ul>
                        <li>K-Means Clustering</li>
                        <li>PCA & t-SNE</li>
                        <li>Anomaly Detection</li>
                        <li>Dimensionality Reduction</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Deep Learning</h3>
                    <ul>
                        <li>Neural Networks</li>
                        <li>CNN for Images</li>
                        <li>RNN/LSTM</li>
                        <li>Transformers</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 NLP</h3>
                    <ul>
                        <li>Text Preprocessing</li>
                        <li>Sentiment Analysis</li>
                        <li>Named Entity Recognition</li>
                        <li>Text Classification</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Time Series</h3>
                    <ul>
                        <li>ARIMA Models</li>
                        <li>Prophet Forecasting</li>
                        <li>LSTM Time Series</li>
                        <li>Trend Analysis</li>
                    </ul>
                </div>
                <div class="feature-card" onclick="toggleCard(this)">
                    <h3>🔹 Recommendation Systems</h3>
                    <ul>
                        <li>Collaborative Filtering</li>
                        <li>Content-Based</li>
                        <li>Matrix Factorization</li>
                        <li>Hybrid Systems</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="content-section" id="quickstart">
            <h2>🚀 Quick Start Guide</h2>
            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>📋 Prerequisites</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <ul>
                        <li>🐍 Python 3.8+ installed</li>
                        <li>📓 Jupyter Notebook or JupyterLab</li>
                        <li>📦 pip or conda package manager</li>
                        <li>💻 Basic Python knowledge</li>
                    </ul>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>⚡ Installation Steps</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <pre><code># Clone the repository
git clone https://github.com/Kiran-id10/data-science-ml-dl-models.git

# Navigate to directory
cd data-science-ml-dl-models

# Create virtual environment
python -m venv venv

# Activate environment
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook</code></pre>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🎯 Using Conda</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <pre><code># Create conda environment
conda create -n datascience python=3.9

# Activate environment
conda activate datascience

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter

# For deep learning
conda install tensorflow pytorch</code></pre>
                </div>
            </div>
        </div>

        <div class="content-section" id="projects">
            <h2>💡 Featured Projects</h2>
            <div class="highlight-box">
                <h3>🏆 Top Notebooks</h3>
                <p>Click on each project to learn more!</p>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>📊 Customer Segmentation</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <p><strong>Difficulty:</strong> ⭐⭐ Intermediate</p>
                    <p><strong>Description:</strong> K-Means clustering analysis on customer data to identify distinct customer segments for targeted marketing.</p>
                    <p><strong>Libraries:</strong> pandas, sklearn, matplotlib, seaborn</p>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🎬 Movie Recommendation System</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <p><strong>Difficulty:</strong> ⭐⭐⭐ Advanced</p>
                    <p><strong>Description:</strong> Build a collaborative filtering recommendation engine using matrix factorization techniques.</p>
                    <p><strong>Libraries:</strong> pandas, surprise, scipy, numpy</p>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>📈 Stock Price Prediction</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <p><strong>Difficulty:</strong> ⭐⭐⭐ Advanced</p>
                    <p><strong>Description:</strong> LSTM-based deep learning model for time series forecasting of stock prices.</p>
                    <p><strong>Libraries:</strong> tensorflow, keras, pandas, matplotlib</p>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>💬 Sentiment Analysis</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <p><strong>Difficulty:</strong> ⭐⭐ Intermediate</p>
                    <p><strong>Description:</strong> NLP pipeline for sentiment classification using transformers and traditional ML methods.</p>
                    <p><strong>Libraries:</strong> nltk, transformers, sklearn, pandas</p>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🖼️ Image Classification</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <p><strong>Difficulty:</strong> ⭐⭐⭐ Advanced</p>
                    <p><strong>Description:</strong> CNN-based image recognition system with transfer learning capabilities.</p>
                    <p><strong>Libraries:</strong> tensorflow, keras, opencv, numpy</p>
                </div>
            </div>
        </div>

        <div class="content-section">
            <h2>🛠️ Tech Stack</h2>
            <div class="tech-stack">
                <div class="tech-badge tech-python">Python</div>
                <div class="tech-badge tech-numpy">NumPy</div>
                <div class="tech-badge tech-pandas">Pandas</div>
                <div class="tech-badge tech-sklearn">Scikit-Learn</div>
                <div class="tech-badge tech-tensorflow">TensorFlow</div>
                <div class="tech-badge tech-pytorch">PyTorch</div>
            </div>
        </div>

        <div class="content-section">
            <h2>📖 Learning Paths</h2>
            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🌱 For Beginners</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <ol>
                        <li>Start with <strong>01-basics/</strong> to understand Python fundamentals</li>
                        <li>Move to <strong>02-eda/</strong> to learn data exploration</li>
                        <li>Try simple models in <strong>03-supervised-learning/</strong></li>
                        <li>Practice with provided datasets</li>
                        <li>Build your first end-to-end project</li>
                    </ol>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🌿 For Intermediate Learners</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <ol>
                        <li>Explore advanced models in <strong>03-supervised-learning/</strong></li>
                        <li>Dive into <strong>04-unsupervised-learning/</strong></li>
                        <li>Experiment with <strong>06-nlp/</strong> and <strong>07-time-series/</strong></li>
                        <li>Build multiple end-to-end projects</li>
                        <li>Start contributing to the repository</li>
                    </ol>
                </div>
            </div>

            <div class="accordion" onclick="toggleAccordion(this)">
                <div class="accordion-header">
                    <span>🌳 For Advanced Practitioners</span>
                    <span class="accordion-arrow">▼</span>
                </div>
                <div class="accordion-content">
                    <ol>
                        <li>Master <strong>05-deep-learning/</strong> architectures</li>
                        <li>Implement custom models and architectures</li>
                        <li>Optimize hyperparameters and model performance</li>
                        <li>Contribute new notebooks and improvements</li>
                        <li>Mentor other learners in the community</li>
                    </ol>
                </div>
            </div>
        </div>

        <div class="cta-section">
            <h2 style="color: white; margin-bottom: 20px;">Ready to Start Your Journey?</h2>
            <p style="font-size: 1.2em; margin-bottom: 30px;">Clone the repository and start learning today!</p>
            <button class="cta-button" onclick="window.open('https://github.com/Kiran-id10/data-science-machine-learning-and-deep-learning-models', '_blank')">
                🚀 View on GitHub
            </button>
            <button class="cta-button" onclick="copyCommand()">
                📋 Copy Clone Command
            </button>
        </div>

        <footer id="contact">
            <h2>📬 Connect With Me</h2>
            <div class="social-links">
                <div class="social-btn github" onclick="window.open('https://github.com/Kiran-id10', '_blank')">G</div>
                <div class="social-btn linkedin" onclick="window.open('https://linkedin.com/in/your-profile', '_blank')">in</div>
                <div class="social-btn twitter" onclick="window.open('https://twitter.com/your-profile', '_blank')">𝕏</div>
            </div>
            <p style="margin-top: 20px; color: #666;">⭐ Star this repository if you find it helpful!</p>
            <p style="margin-top: 10px; color: #999;">Built with ❤️ for learners and practitioners</p>
            <p style="margin-top: 10px; color: #999;">© 2025 Kiran | MIT License</p>
        </footer>
    </div>

    <script>
        function toggleAccordion(element) {
            element.classList.toggle('active');
        }

        function toggleCard(card) {
            card.style.transform = card.style.transform === 'scale(1.05)' ? 'scale(1)' : 'scale(1.05)';
        }

        function scrollToSection(id) {
            document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
        }

        function copyCommand() {
            const command = 'git clone https://github.com/Kiran-id10/data-science-machine-learning-and-deep-learning-models.git';
            navigator.clipboard.writeText(command).then(() => {
                alert('✅ Clone command copied to clipboard!');
            });
        }

        // Add floating animation on scroll
        window.addEventListener('scroll', () => {
            const emojis = document.querySelectorAll('.floating-emoji');
            emojis.forEach((emoji, index) => {
                const speed = (index + 1) * 0.1;
                emoji.style.transform = `translateY(${window.scrollY * speed}px)`;
            });
        });

        // Add entrance animations to feature cards
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        document.querySelectorAll('.feature-card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'all 0.6s ease';
            observer.observe(card);
        });
    </script>
</body>
</html>
