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

    resultDiv.innerHTML = `
        <h2>Prediction: ${prediction}</h2>

        <p><strong>Confidence:</strong></p>
        <p>Fake: ${(probs.prob_fake * 100).toFixed(2)}%</p>
        <p>Real: ${(probs.prob_real * 100).toFixed(2)}%</p>

        <h3>Why this prediction?</h3>
        <p><strong>Fake signals:</strong> ${
            explanation.top_fake_words.map(w => w[0]).join(", ")
        }</p>

        <p><strong>Real signals:</strong> ${
            explanation.top_real_words.map(w => w[0]).join(", ")
        }</p>
    `;
}