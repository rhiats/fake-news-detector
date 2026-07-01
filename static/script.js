async function analyzeArticle() {
    const text = document.getElementById("articleText").value;
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");

    resultDiv.innerHTML = "";
    loadingDiv.style.display = "block";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        loadingDiv.style.display = "none";

        displayResult(data);

    } catch (error) {
        loadingDiv.style.display = "none";
        resultDiv.innerHTML = "Error";
        console.error(error);
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById("result");

    const prediction = data.prediction;
    const probs = data.probabilities;

    const fakePct = (probs.prob_fake * 100).toFixed(1);
    const realPct = (probs.prob_real * 100).toFixed(1);

    const color = prediction === "Fake News" ? "#ef4444" : "#22c55e";

    resultDiv.innerHTML = `
        <div class="fade-in">

            <h2 style="color:${color}">
                ${prediction}
            </h2>

            <h3>Confidence</h3>

            <div class="bar-container">
                <div class="bar fake" style="width:${fakePct}%">
                    Fake ${fakePct}%
                </div>
            </div>

            <div class="bar-container">
                <div class="bar real" style="width:${realPct}%">
                    Real ${realPct}%
                </div>
            </div>

        </div>
    `;

    document.getElementById("feedback").style.display = "block";

    window.lastPrediction = prediction === "Fake News" ? 1 : 0;
}

async function sendFeedback(trueLabel) {

    const text = window.currentArticle;
    const predictedLabel = window.lastPrediction;

    console.log("Sending:", text, trueLabel, predictedLabel);

    await fetch("/feedback", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text,
            true_label: trueLabel,
            predicted_label: predictedLabel
        })
    });

    alert("Feedback saved");
}