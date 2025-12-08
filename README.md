<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science & ML Repository - Interactive Guide</title>
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
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 60px 20px;
            background: rgba(255, 255, 255, 0.98);
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
            font-size: 3.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }

        .subtitle {
            font-size: 1.4em;
            color: #555;
            margin: 15px 0;
        }

        .badges {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin: 25px 0;
        }

        .badge {
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.95em;
            font-weight: bold;
            transition: all 0.3s;
            cursor: pointer;
        }

        .badge:hover {
            transform: translateY(-5px) scale(1.05);
        }

        .badge-notebooks { background: #FF6B6B; color: white; }
        .badge-python { background: #3776AB; color: white; }
        .badge-ml { background: #4CAF50; color: white; }
        .badge-dl { background: #E91E63; color: white; }
        .badge-stats { background: #9C27B0; color: white; }

        .search-container {
            margin: 30px 0;
            text-align: center;
        }

        .search-box {
            width: 60%;
            padding: 15px 25px;
            font-size: 1.1em;
            border: 3px solid #667eea;
            border-radius: 50px;
            outline: none;
            transition: all 0.3s;
        }

        .search-box:focus {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            transform: scale(1.02);
        }

        .filter-tabs {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin: 20px 0;
        }

        .filter-tab {
            padding: 12px 25px;
            background: white;
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }

        .filter-tab:hover, .filter-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: scale(1.05);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: all 0.3s;
            cursor: pointer;
        }

        .stat-card:hover {
            transform: translateY(-10px) rotate(2deg);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }

        .stat-number {
            font-size: 3.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            color: #666;
            margin-top: 10px;
            font-size: 1.2em;
            font-weight: 600;
        }

        .content-section {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h2 {
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .notebook-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .notebook-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.4s;
            border: 3px solid transparent;
            position: relative;
            overflow: hidden;
        }

        .notebook-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
            transition: 0.5s;
        }

        .notebook-card:hover::before {
            left: 100%;
        }

        .notebook-card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            border-color: #667eea;
        }

        .notebook-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .notebook-title {
            color: #667eea;
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .notebook-category {
            display: inline-block;
            padding: 5px 15px;
            background: #667eea;
            color: white;
            border-radius: 15px;
            font-size: 0.85em;
            margin: 5px 0;
        }

        .notebook-description {
            color: #555;
            margin-top: 10px;
            line-height: 1.6;
        }

        .difficulty {
            margin-top: 10px;
            font-weight: bold;
        }

        .difficulty-easy { color: #4CAF50; }
        .difficulty-medium { color: #FF9800; }
        .difficulty-hard { color: #F44336; }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            animation: fadeIn 0.3s;
        }

        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 40px;
            border-radius: 20px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            animation: slideIn 0.3s;
        }

        @keyframes slideIn {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .close {
            color: #aaa;
            float: right;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        .close:hover {
            color: #667eea;
            transform: scale(1.2);
        }

        .quick-start {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin: 30px 0;
        }

        .quick-start h3 {
            font-size: 2em;
            margin-bottom: 20px;
        }

        .steps {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .step {
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            transition: all 0.3s;
        }

        .step:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }

        .step-number {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        code {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 10px;
            display: block;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }

        .cta-section {
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            color: white;
            margin: 30px 0;
        }

        .cta-button {
            padding: 18px 45px;
            background: white;
            color: #667eea;
            border: none;
            border-radius: 35px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .cta-button:hover {
            transform: scale(1.1) rotate(-2deg);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }

        .learning-path {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
            margin: 30px 0;
        }

        .path-card {
            flex: 1;
            min-width: 300px;
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }

        .path-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }

        .path-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .path-title {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 15px;
        }

        .floating-emoji {
            position: fixed;
            font-size: 3em;
            animation: float 3s ease-in-out infinite;
            pointer-events: none;
            z-index: 999;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(10deg); }
        }

        footer {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            margin-top: 30px;
        }

        .progress-bar {
            position: fixed;
            top: 0;
            left: 0;
            height: 5px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            z-index: 9999;
            transition: width 0.3s;
        }

        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            h2 { font-size: 1.8em; }
            .notebook-grid { grid-template-columns: 1fr; }
            .search-box { width: 90%; }
        }
    </style>
</head>
<body>
    <div class="progress-bar" id="progressBar"></div>
    
    <div class="floating-emoji" style="left: 5%; top: 10%;">🚀</div>
    <div class="floating-emoji" style="right: 5%; top: 20%; animation-delay: 1s;">📊</div>
    <div class="floating-emoji" style="left: 10%; bottom: 20%; animation-delay: 2s;">🤖</div>
    <div class="floating-emoji" style="right: 10%; bottom: 15%; animation-delay: 1.5s;">🧠</div>

    <div class="container">
        <header>
            <h1>🎓 Data Science & Machine Learning Repository</h1>
            <p class="subtitle">Your Complete Learning Journey from Python Basics to Deep Learning</p>
            
            <div class="badges">
                <span class="badge badge-notebooks">20 Notebooks</span>
                <span class="badge badge-python">Python</span>
                <span class="badge badge-ml">Machine Learning</span>
                <span class="badge badge-dl">Deep Learning</span>
                <span class="badge badge-stats">Statistics</span>
            </div>

            <div class="search-container">
                <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search notebooks (e.g., 'regression', 'neural', 'clustering')..." onkeyup="searchNotebooks()">
            </div>

            <div class="filter-tabs">
                <div class="filter-tab active" onclick="filterNotebooks('all')">All (20)</div>
                <div class="filter-tab" onclick="filterNotebooks('basics')">Basics (3)</div>
                <div class="filter-tab" onclick="filterNotebooks('statistics')">Statistics (2)</div>
                <div class="filter-tab" onclick="filterNotebooks('ml')">ML Algorithms (9)</div>
                <div class="filter-tab" onclick="filterNotebooks('dl')">Deep Learning (2)</div>
                <div class="filter-tab" onclick="filterNotebooks('advanced')">Advanced (4)</div>
            </div>
        </header>

        <div class="content-section">
            <h2>📊 Repository Overview</h2>
            <div class="stats-grid">
                <div class="stat-card" onclick="filterNotebooks('basics')">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Python Basics</div>
                </div>
                <div class="stat-card" onclick="filterNotebooks('ml')">
                    <div class="stat-number">9</div>
                    <div class="stat-label">ML Models</div>
                </div>
                <div class="stat-card" onclick="filterNotebooks('dl')">
                    <div class="stat-number">2</div>
                    <div class="stat-label">Deep Learning</div>
                </div>
                <div class="stat-card" onclick="filterNotebooks('advanced')">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Advanced Topics</div>
                </div>
            </div>
        </div>

        <div class="content-section">
            <h2>📚 All Notebooks</h2>
            <div class="notebook-grid" id="notebookGrid">
                <!-- Python Basics -->
                <div class="notebook-card" data-category="basics" onclick="showDetails('basics-python')">
                    <div class="notebook-icon">🐍</div>
                    <div class="notebook-title">Basics of Python</div>
                    <span class="notebook-category">Fundamentals</span>
                    <div class="difficulty difficulty-easy">⭐ Beginner</div>
                    <div class="notebook-description">Core Python concepts, syntax, and programming fundamentals</div>
                </div>

                <div class="notebook-card" data-category="basics" onclick="showDetails('data-structures')">
                    <div class="notebook-icon">📦</div>
                    <div class="notebook-title">Python Data Structures</div>
                    <span class="notebook-category">Fundamentals</span>
                    <div class="difficulty difficulty-easy">⭐ Beginner</div>
                    <div class="notebook-description">Lists, tuples, dictionaries, sets, and data manipulation</div>
                </div>

                <div class="notebook-card" data-category="basics" onclick="showDetails('data-transformation')">
                    <div class="notebook-icon">🔄</div>
                    <div class="notebook-title">Data Transformation</div>
                    <span class="notebook-category">Data Processing</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Feature scaling, encoding, normalization, and preprocessing</div>
                </div>

                <!-- Statistics -->
                <div class="notebook-card" data-category="statistics" onclick="showDetails('basic-stats-1')">
                    <div class="notebook-icon">📊</div>
                    <div class="notebook-title">Basic Statistics 1</div>
                    <span class="notebook-category">Statistics</span>
                    <div class="difficulty difficulty-easy">⭐ Beginner</div>
                    <div class="notebook-description">Mean, median, mode, variance, standard deviation</div>
                </div>

                <div class="notebook-card" data-category="statistics" onclick="showDetails('basic-stats-2')">
                    <div class="notebook-icon">📈</div>
                    <div class="notebook-title">Basic Statistics 2</div>
                    <span class="notebook-category">Statistics</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Distributions, correlation, probability concepts</div>
                </div>

                <div class="notebook-card" data-category="statistics" onclick="showDetails('hypothesis-testing')">
                    <div class="notebook-icon">🔬</div>
                    <div class="notebook-title">Hypothesis Testing</div>
                    <span class="notebook-category">Statistics</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">T-tests, ANOVA, Chi-square, p-values, statistical significance</div>
                </div>

                <!-- EDA -->
                <div class="notebook-card" data-category="basics" onclick="showDetails('eda')">
                    <div class="notebook-icon">🔍</div>
                    <div class="notebook-title">Exploratory Data Analysis</div>
                    <span class="notebook-category">Data Analysis</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Data visualization, patterns, outliers, univariate & multivariate analysis</div>
                </div>

                <!-- ML Algorithms -->
                <div class="notebook-card" data-category="ml" onclick="showDetails('linear-regression')">
                    <div class="notebook-icon">📉</div>
                    <div class="notebook-title">Multiple Linear Regression</div>
                    <span class="notebook-category">Supervised Learning</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Linear models, feature selection, R-squared, residual analysis</div>
                </div>

                <div class="notebook-card" data-category="ml" onclick="showDetails('logistic-regression')">
                    <div class="notebook-icon">🎯</div>
                    <div class="notebook-title">Logistic Regression</div>
                    <span class="notebook-category">Classification</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Binary classification, sigmoid function, odds ratio</div>
                </div>

                <div class="notebook-card" data-category="ml" onclick="showDetails('decision-tree')">
                    <div class="notebook-icon">🌳</div>
                    <div class="notebook-title">Decision Tree</div>
                    <span class="notebook-category">Classification</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">Tree-based models, entropy, gini index, pruning</div>
                </div>

                <div class="notebook-card" data-category="ml" onclick="showDetails('random-forest')">
                    <div class="notebook-icon">🌲</div>
                    <div class="notebook-title">Random Forest</div>
                    <span class="notebook-category">Ensemble Learning</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Ensemble methods, bagging, feature importance</div>
                </div>

                <div class="notebook-card" data-category="ml" onclick="showDetails('svm')">
                    <div class="notebook-icon">🎪</div>
                    <div class="notebook-title">Support Vector Machine</div>
                    <span class="notebook-category">Classification</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Kernel methods, margin optimization, SVC & SVR</div>
                </div>

                <div class="notebook-card" data-category="ml" onclick="showDetails('xgboost')">
                    <div class="notebook-icon">⚡</div>
                    <div class="notebook-title">XGBoost & LightGBM</div>
                    <span class="notebook-category">Gradient Boosting</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Advanced boosting algorithms, hyperparameter tuning</div>
                </div>

                <!-- Unsupervised Learning -->
                <div class="notebook-card" data-category="ml" onclick="showDetails('clustering')">
                    <div class="notebook-icon">🔵</div>
                    <div class="notebook-title">Clustering</div>
                    <span class="notebook-category">Unsupervised Learning</span>
                    <div class="difficulty difficulty-medium">⭐⭐ Intermediate</div>
                    <div class="notebook-description">K-Means, hierarchical clustering, DBSCAN</div>
                </div>

                <div class="notebook-card" data-category="advanced" onclick="showDetails('pca')">
                    <div class="notebook-icon">📐</div>
                    <div class="notebook-title">Principal Component Analysis</div>
                    <span class="notebook-category">Dimensionality Reduction</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Feature reduction, variance explained, eigenvectors</div>
                </div>

                <!-- Deep Learning -->
                <div class="notebook-card" data-category="dl" onclick="showDetails('neural-networks')">
                    <div class="notebook-icon">🧠</div>
                    <div class="notebook-title">Neural Networks</div>
                    <span class="notebook-category">Deep Learning</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Feedforward networks, backpropagation, activation functions</div>
                </div>

                <div class="notebook-card" data-category="dl" onclick="showDetails('rnn')">
                    <div class="notebook-icon">🔁</div>
                    <div class="notebook-title">Recurrent Neural Networks</div>
                    <span class="notebook-category">Deep Learning</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">LSTM, GRU, sequence modeling, time dependencies</div>
                </div>

                <!-- Advanced Topics -->
                <div class="notebook-card" data-category="advanced" onclick="showDetails('nlp')">
                    <div class="notebook-icon">💬</div>
                    <div class="notebook-title">NLP Sentiment Analysis</div>
                    <span class="notebook-category">Natural Language Processing</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Text processing, tokenization, sentiment classification</div>
                </div>

                <div class="notebook-card" data-category="advanced" onclick="showDetails('time-series')">
                    <div class="notebook-icon">📅</div>
                    <div class="notebook-title">Time Series Forecasting</div>
                    <span class="notebook-category">Time Series</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">ARIMA, seasonality, trend analysis, forecasting</div>
                </div>

                <div class="notebook-card" data-category="advanced" onclick="showDetails('recommendation')">
                    <div class="notebook-icon">⭐</div>
                    <div class="notebook-title">Recommendation System</div>
                    <span class="notebook-category">Recommendation Systems</span>
                    <div class="difficulty difficulty-hard">⭐⭐⭐ Advanced</div>
                    <div class="notebook-description">Collaborative filtering, content-based, matrix factorization</div>
                </div>
            </div>
        </div>

        <div class="quick-start">
            <h3>🚀 Quick Start Guide</h3>
            <div class="steps">
                <div class="step">
                    <div class="step-number">1</div>
                    <h4>Clone Repository</h4>
                    <p>Get the code on your machine</p>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <h4>Install Dependencies</h4>
                    <p>Set up your environment</p>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <h4>Launch Jupyter</h4>
                    <p>Open and run notebooks</p>
                </div>
                <div class="step">
                    <div class="step-number">4</div>
                    <h4>Start Learning</h4>
                    <p>Follow the learning path</p>
                </div>
            </div>
            <code>git clone https://github.com/Kiran-id10/data-science-ml-dl-models.git
cd data-science-ml-dl-models
pip install -r requirements.txt
jupyter notebook</code>
        </div>

        <div class="content-section">
            <h2>🎯 Recommended Learning Paths</h2>
            <div class="learning-path">
                <div class="path-card">
                    <div class="path-icon">🌱</div>
                    <div class="path-title">Beginner Path</div>
                    <ol style="text-align: left; color: #666;">
                        <li>Basics of Python</li>
                        <li>Python Data Structures</li>
                        <li>Basic Statistics 1 & 2</li>
                        <li>Exploratory Data Analysis</li>
                        <li>Linear Regression</li>
                        <li>Logistic Regression</li>
                    </ol>
                </div>
                <div class="path-card">
                    <div class="path-icon">🌿</div>
                    <div class="path-title">Intermediate Path</div>
                    <ol style="text-align: left; color: #666;">
                        <li>Data Transformation</li>
                        <li>Hypothesis Testing</li>
                        <li>Decision Tree</li>
                        <li>Random Forest</li>
                        <li>Clustering</li>
                        <li>Support Vector Machine</li>
                    </ol>
                </div>
                <div class="path-card">
                    <div class="path-icon">🌳</div>
                    <div class="path-title">Advanced Path</div>
                    <ol
