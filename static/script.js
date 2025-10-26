// Browser based inference using Tensorflow.js
async function runModel() {
    const model = await tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [4], units: 8 , activation: 'relu'}),
            tf.layers.dense({units:1, activation: 'sigmoid'})
        ]
    });

    const_input = tf.tensor2d([[0.5,0.2,0.1,0.7]]);
    const_output = model.predict(input);
    const value = await output.data();

    document.getElementById("tf-result").innerText = 
        "Tensorflow.js output (in browser): " + value[0].toFixed(4);
}

// OpenAI text generation 
async function askOpenAI() {
    const prompt = document.getElementById("prompt").value;

    const response = await fetch("/generate_text", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: prompt})
    });

    const data = await response.json();
    document.getElementById("ai-response").innerText = "💬 " + data.response;
}