<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Notebook Interactive Dashboard</title>
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea, #764ba2);
        margin: 0;
        padding: 0;
        color: #333;
    }

    .container {
        max-width: 1300px;
        margin: auto;
        padding: 20px;
    }

    header {
        background: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }

    h1 {
        font-size: 3.2em;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .subtitle {
        color: #555;
        font-size: 1.3em;
        margin-bottom: 20px;
    }

    .search-box {
        width: 60%;
        padding: 15px;
        font-size: 1.2em;
        border-radius: 40px;
        border: 3px solid #667eea;
        outline: none;
    }

    .filter-tabs {
        margin: 20px;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
    }

    .filter-tab {
        padding: 12px 25px;
        border-radius: 25px;
        border: 2px solid #fff;
        color: white;
        background: rgba(255,255,255,0.2);
        cursor: pointer;
        font-weight: bold;
        transition: 0.3s;
    }

    .filter-tab:hover, .filter-tab.active {
        background: white;
        color: #667eea;
    }

    .notebook-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin-top: 30px;
    }

    .notebook-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: 0.3s;
        cursor: pointer;
    }

    .notebook-card:hover {
        transform: translateY(-10px);
    }

    .notebook-title {
        color: #667eea;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .notebook-category {
        background: #667eea;
        color: white;
        padding: 5px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin-bottom: 10px;
    }

    .notebook-description {
        color: #555;
        line-height: 1.5;
    }

</style>

<script>
function filterCategory(category) {
    const cards = document.querySelectorAll('.notebook-card');
    cards.forEach(card => {
        if (category === 'all' || card.dataset.category === category) {
            card.style.display = "block";
        } else {
            card.style.display = "none";
        }
    });

    const tabs = document.querySelectorAll('.filter-tab');
    tabs.forEach(t => t.classList.remove('active'));
    document.getElementById(category).classList.add('active');
}

function searchNotebooks() {
    let term = document.getElementById('search').value.toLowerCase();
    let cards = document.querySelectorAll('.notebook-card');

    cards.forEach(card => {
        let text = card.innerText.toLowerCase();
        card.style.display = text.includes(term) ? "block" : "none";
    });
}
</script>
</head>

<body>
<div class="container">

<header>
    <h1>📘 ML Notebook Dashboard</h1>
    <p class="subtitle">Explore Machine Learning, Deep Learning, Python & Statistics Notebooks</p>
    <input type="text" id="search" class="search-box" placeholder="Search notebooks..." onkeyup="searchNotebooks()">
</header>

<div class="filter-tabs">
    <div class="filter-tab active" id="all" onclick="filterCategory('all')">All</div>
    <div class="filter-tab" id="python" onclick="filterCategory('python')">Python</div>
    <div class="filter-tab" id="ml" onclick="filterCategory('ml')">Machine Learning</div>
    <div class="filter-tab" id="dl" onclick="filterCategory('dl')">Deep Learning</div>
    <div class="filter-tab" id="stats" onclick="filterCategory('stats')">Statistics</div>
</div>

<div class="notebook-grid">

    <div class="notebook-card" data-category="python">
        <div class="notebook-title">Python Basics</div>
        <div class="notebook-category">Python</div>
        <div class="notebook-description">Covers variables, loops, functions, and essential Python foundations.</div>
    </div>

    <div class="notebook-card" data-category="ml">
        <div class="notebook-title">Linear Regression</div>
        <div class="notebook-category">Machine Learning</div>
        <div class="notebook-description">Complete guide to linear regression with sklearn, plots, and metrics.</div>
    </div>

    <div class="notebook-card" data-category="dl">
        <div class="notebook-title">Image Classification (CNN)</div>
        <div class="notebook-category">Deep Learning</div>
        <div class="notebook-description">Build a CNN model using TensorFlow/Keras for image recognition.</div>
    </div>

    <div class="notebook-card" data-category="stats">
        <div class="notebook-title">Hypothesis Testing</div>
        <div class="notebook-category">Statistics</div>
        <div class="notebook-description">Z-test, T-test, ANOVA, Chi-square tests with Python examples.</div>
    </div>

</div>

</div>
</body>
</html>
