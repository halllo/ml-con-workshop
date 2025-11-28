//copy this into browser console at http://localhost:50602/
async function predict() {
  for (let i = 0; i < 100; i++) {
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({"request": {"text": "Go is amazing!","request_id": null}})
      });
      
      const data = await response.json();
      console.log(`Request ${i + 1}:`, data);
    } catch (error) {
      console.error(`Error on request ${i + 1}:`, error);
    }
  }
}

predict();