document.getElementById('solveBtn').addEventListener('click', async () => {
    const status = document.getElementById('status');
    status.innerText = "Analyzing board...";

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab) {
        chrome.tabs.sendMessage(tab.id, { action: "solve" }, (response) => {
            if (chrome.runtime.lastError) {
                status.innerText = "Error: Please refresh the game page.";
                console.error(chrome.runtime.lastError);
            } else {
                status.innerText = "Done! Check the game screen.";
            }
        });
    }
});
