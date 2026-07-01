async function analyzeArticle() {
    const text = document.getElementById("articleText").value;
    const resultDiv = document.getElementById("result");

    // show loading state
    resultDiv.innerHTML = "Analyzing... 🧠";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        console.log(data); // always inspect first

        displayResult(data);

    } catch (error) {
        resultDiv.innerHTML = "Error connecting to server ❌";
        console.error(error);
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById("result");

    const prediction = data.prediction;
    const probs = data.probabilities;
    const explanation = data.explanation;

    let color = prediction === "Fake News" ? "#ef4444" : "#22c55e";

    resultDiv.innerHTML = `
        <h2 style="color:${color}">
            ${prediction === "Fake News" ? "🚨 Fake News" : "✅ Real News"}
        </h2>

        <p><strong>Confidence Scores</strong></p>

        <div class="bar-container">
            <div class="bar fake" style="width:${probs.prob_fake * 100}%">
                Fake ${(probs.prob_fake * 100).toFixed(1)}%
            </div>
        </div>

        <div class="bar-container">
            <div class="bar real" style="width:${probs.prob_real * 100}%">
                Real ${(probs.prob_real * 100).toFixed(1)}%
            </div>
        </div>

        <h3>Why this prediction?</h3>

        <p><strong>Fake signals:</strong> ${
            explanation.top_fake_words.map(w => w[0]).join(", ")
        }</p>

        <p><strong>Real signals:</strong> ${
            explanation.top_real_words.map(w => w[0]).join(", ")
        }</p>
    `;
}